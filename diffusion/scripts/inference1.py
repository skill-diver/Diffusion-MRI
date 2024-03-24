# 但是脑部mri有中间空的区域，如果只以矩形去限制平移有时候会出错，有什么确保不会出错的办法
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
from visdom import Visdom
viz = Visdom(port=8850) # 初始化Visdom可视化界面，port为8850
import sys
 # 添加当前目录到系统路径
sys.path.append("..")
sys.path.append(".")
from guided_diffusion.bratsloader import BRATSDataset
import torch.nn.functional as F
import numpy as np
import torch as th
import torch.distributed as dist
from guided_diffusion.image_datasets import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_classifier,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def compute_gradient(image):
    rows, cols = image.shape
    gradient_image = np.zeros((rows, cols))
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            dx = abs(image[i, j] - image[i, j - 1])
            dy = abs(image[i, j] - image[i - 1, j])
            
            gradient_magnitude = np.sqrt(dx ** 2 + dy ** 2)
            gradient_image[i, j] = gradient_magnitude
            
    return gradient_image

def random_translate_diff(diff, img_shape, area):
    print('fid',diff)
    print('ima',img_shape)
    print('ar',area)
    non_zero_coords = diff > 0.2
    print(non_zero_coords.shape)
    
    # min_y, max_y = np.min(non_zero_coords[0]), np.max(non_zero_coords[0])
    # min_x, max_x = np.min(non_zero_coords[1]), np.max(non_zero_coords[1])
    min_y, max_y = np.min(non_zero_coords[0]), np.max(non_zero_coords[0])
    min_x, max_x = non_zero_coords[1][0], non_zero_coords[1][-1]
    
    print("non_zero_coords[0]",non_zero_coords)
    print("non_zero_coords[1]",non_zero_coords[1])

    diff_height = max_y - min_y + 1
    diff_width = max_x - min_x + 1
    
    max_translate_y = img_shape[2][-1] - img_shape[2][0] - diff_height
    max_translate_x = img_shape[3][-1] - img_shape[3][0] - diff_width
    
    
    print("img_shape[3]",img_shape[3])
    print("img_shape[2]",img_shape[2])
    
    x = 0 if max_translate_x <= 0 else np.random.randint(img_shape[3][0], max_translate_x)
    y = 0 if max_translate_y <= 0 else np.random.randint(img_shape[2][0], max_translate_y)
    
    new_diff = np.zeros(area)
    new_diff[:, :, y:y+diff_height, x:x+diff_width] = diff[min_y:max_y+1, min_x:max_x+1]

    return new_diff



"""
main函数:
    功能: 
        - 根据命令行参数设置模型，加载数据集，加载分类器并进行样本生成。
    输入: 
        - 命令行参数 (通过argparse库解析)
    输出: 
        - 样本，原始图像，标签和弱标签都保存在一个.npz文件中。
"""
def main():
    # 解析从命令行传入的参数
    args = create_argparser().parse_args()
    # 设置分布式计算和日志配置
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # 加载数据
    ds = BRATSDataset(args.data_dir, test_flag=True)
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    # 加载模型
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    # 加载分类器
    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path)
    )
    # 打印加载的分类器信息和参数数量
    print('loaded classifier')
    p1 = np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()
    p2 = np.array([np.array(p.shape).prod() for p in classifier.parameters()]).sum()
    print('pmodel', p1, 'pclass', p2)
    # 将分类器转移到适当的设备并设置为评估模式
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()
    # 定义条件函数和模型函数
    def cond_fn(x, t, y=None, gt=None, gt_keep_mask=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            a = th.autograd.grad(selected.sum(), x_in)[0]
            return a, a * args.classifier_scale

    def model_fn(x, t, y=None, gt=None, gt_keep_mask=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    # The sampling process
    logger.log("sampling...")
    #mask_range = []
    samples = []
    orgs = []
    labels = []
    weak_labels = []
    #moved_masks = []  # 用于存储移动后的masks
    diffs = []
    i = 0
    # 从文件中加载mask并转换为tensor格式
    pil_mask = Image.open(args.mask_path)
    tensor_mask = th.from_numpy(np.array(pil_mask)).float().to(dist_util.dev())
    batch_idx = 0
    # 遍历数据集中的图像
    for img in datal:
        
        # 初始化一个字典来保存模型的关键字参数
        model_kwargs = {}
        #print('img', img[0].shape, img[1])
        # 创建一个mask来识别图像中的大脑部分
        brain_mask = np.where(img[0] > 0, 1, 0)
        brain_mask_single_channel = th.from_numpy(np.any(brain_mask, axis=1)).float().to(dist_util.dev())
        brain_mask_multi_channel = brain_mask_single_channel.unsqueeze(1).repeat(1, 4, 1, 1)
        # 获取大脑图像的非零坐标，即大脑部分的位置
        non_zero_coords = np.array(np.where(brain_mask))
     
        # 如果数据集是"brats"，对图像进行进一步处理
        if args.dataset == 'brats':
            Labelmask = th.where(img[3] > 0, 1, 0)
            number = img[4][0]
            # if img[2] == 0:
            #     continue
            if all(element == 1 for element in img[2]):
                continue

        # 如果模型需要类条件，为其设置对应的类标签和其他关键参数
        if args.class_cond:
            # unhealthy -> healthy
            # classes = th.randint(
            #   low=0, high=1, size=(args.batch_size,), device=dist_util.dev()
            # )
            #healthy->unhealthy
            classes = th.randint(
                low=1, high=2, size=(args.batch_size,), device=dist_util.dev()
            )

            # Initialization of model_kwargs. This is used for model conditioning.
            # y: target class as weak labels.
            # gt: original image as the repaint pixel-wise conditioning
            # gt_keep_mask: mask indicating where to generate new partition of the image.
            model_kwargs["y"] = classes
            model_kwargs['gt'] = img[0].to(dist_util.dev())
            model_kwargs['gt_keep_mask'] = tensor_mask * brain_mask_multi_channel

            #print('y', model_kwargs["y"])
        # 根据是否使用ddim选择合适的样本生成函数
        sample_fn = (
            diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
        )
        #print('samplefn', sample_fn)
        # 使用所选的函数生成样本
        sample, x_noisy, org = sample_fn(
            model_fn,
            (args.batch_size, 4, args.image_size, args.image_size), img, org=img,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
            noise_level=args.noise_level,
            use_resample=args.use_resample
        )
        #mask_range.append(non_zero_coords)
        samples.append(sample.cpu().numpy())
        orgs.append(org.cpu().numpy())
        labels.append(img[3].cpu().numpy())
        weak_labels.append(img[2].cpu().numpy())
        # print(f"Length of samples: {len(samples)}")
        # print(f"Length of orgs: {len(orgs)}")
        # print(f"Length of labels: {len(labels)}")
        # print(f"Length of weak_labels: {len(weak_labels)}")
        
        
        samples_array = np.concatenate(samples, axis=0)
        orgs_array = np.concatenate(orgs, axis=0)
        labels_array = np.concatenate(labels, axis=0)
        weak_labels_array = np.concatenate(weak_labels, axis=0)
        print("sample array shape", samples_array.shape)
        diffs = []
        #diffs_distribution = []
        # sample 是推理后的结果
        print("shape of sample:", samples_array.shape)
        #print("sample:", samples_array)
        print("shape of orgs:", orgs_array.shape)
        current_diff = np.sum(abs(samples_array[0] - orgs_array[0]), axis=0)
        mask_area = np.where(orgs_array > 0)    #
        #print("shape of mask_area:", mask_area.shape)
        #mask_non_zero_coords = np.array(np.where(mask_area))
        #print("Shape of mask_non_zero_coords:", mask_non_zero_coords.shape)

        print("Shape of diff:", current_diff.shape)
        translated_diff = random_translate_diff(current_diff,  mask_area, img[0].shape)
            
        # 将平移后的差异图像添加到diffs列表中。
        diffs.append(translated_diff)
            
        diffs_array = np.stack(diffs, axis=0)
        save_path = f"{args.sample_path}_batch_{batch_idx}.npz"
        np.savez(save_path,
         samples=samples_array,
         orgs=orgs_array,
         labels=labels_array,
         weak_labels=weak_labels_array,
         translated_diffs=diffs_array
        )

        samples.clear()
        orgs.clear()
        labels.clear()
        weak_labels.clear()
        # samples_array.clear()
        # orgs_array.clear()
        # labels_array.clear()
        # weak_labels_array.clear()
        # diffs_array.clear()
        batch_idx += 1
        i += 1
   




"""
create_argparser函数:
    功能: 
        - 创建一个用于解析命令行参数的解析器。
    输入: 
        - 无
    输出: 
        - argparse.ArgumentParser对象，用于解析命令行参数。
"""
def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_samples=5000,
        batch_size=32,   ###1
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=100,
        noise_level=500,
        dataset='brats',
        mask_path="",
        use_resample=False,
        n_generated_batch=5,
        sample_path=""
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()