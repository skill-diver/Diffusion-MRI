from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
from visdom import Visdom
viz = Visdom(port=8850) 
import sys

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

def generate_new_mask(original_mask):
    # 随机选择一个包含肿瘤的区域
    tumor_coords = np.where(original_mask > 0)
    min_y, max_y = min(tumor_coords[0]), max(tumor_coords[0])
    min_x, max_x = min(tumor_coords[1]), max(tumor_coords[1])

    roi = original_mask[min_y:max_y, min_x:max_x]

    # 随机缩放ROI
    scale_factor = np.random.uniform(0.5, 1.5)  # 随机选择一个缩放factor
    new_shape = (int(roi.shape[0] * scale_factor), int(roi.shape[1] * scale_factor))
    roi_rescaled = cv2.resize(roi, new_shape)

    # 随机选择一个新位置
    start_y = np.random.randint(0, original_mask.shape[0] - roi_rescaled.shape[0])
    start_x = np.random.randint(0, original_mask.shape[1] - roi_rescaled.shape[1])

    # 创建一个新的空mask并将缩放后的ROI粘贴到随机位置
    new_mask = np.zeros_like(original_mask)
    new_mask[start_y:start_y+roi_rescaled.shape[0], start_x:start_x+roi_rescaled.shape[1]] = roi_rescaled

    return new_mask
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

    samples = []
    orgs = []
    labels = []
    weak_labels = []
    i = 0

    pil_mask = Image.open(args.mask_path)
    tensor_mask = th.zeros((args.batch_size, 1, args.image_size, args.image_size), dtype=th.float32).to(dist_util.dev())
    for img in datal:  # 遍历数据加载器datal中的每一批图像
        if i >= args.n_generated_batch:  # 如果生成的批次数量达到了指定的上限，跳出循环
            break

        model_kwargs = {}  # 初始化一个空的模型参数字典

        # 打印当前批次的图像形状和标签
        print('img', img[0].shape, img[1])

        # 创建大脑mask，大于0的值为1，否则为0
        brain_mask = np.where(img[0] > 0, 1, 0)
        
        # 将多通道的大脑mask转换为单通道
        brain_mask_single_channel = th.from_numpy(np.any(brain_mask, axis=1)).float().to(dist_util.dev())
        
        # 通过复制单通道大脑mask，创建多通道版本
        brain_mask_multi_channel = brain_mask_single_channel.unsqueeze(1).repeat(1, 4, 1, 1)
        print('brain_mask shape', brain_mask_single_channel.shape)

        # 对于brats数据集的处理
        if args.dataset == 'brats':
            # 创建一个label mask，其中大于0的值为1，否则为0
            Labelmask = th.where(img[3] > 0, 1, 0)
            
            # 获取当前批次的图像编号
            number = img[4][0]
            
            # 如果所有图像的元素都为1，则继续下一个批次
            if all(element == 1 for element in img[2]):
                continue

        # 条件模型的处理
        if args.class_cond:
            # 生成随机类，这里固定生成类1
            classes = th.randint(
                low=1, high=2, size=(args.batch_size,), device=dist_util.dev()
            )

            # 设置模型参数，其中y为目标类，gt为原始图像，gt_keep_mask为指示图像生成部分的mask
            model_kwargs["y"] = classes
            model_kwargs['gt'] = img[0].to(dist_util.dev())
            model_kwargs['gt_keep_mask'] = tensor_mask * brain_mask_multi_channel

            print('y', model_kwargs["y"])

        # 选择合适的样本函数，基于是否使用DDIM
        sample_fn = (
            diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
        )
        print('samplefn', sample_fn)

        # 调用样本函数，生成样本
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

        # 将生成的样本和原始数据添加到列表中
        samples.append(sample.cpu().numpy())
        orgs.append(org.cpu().numpy())
        labels.append(img[3].cpu().numpy())
        weak_labels.append(img[2].cpu().numpy())
        
        i += 1  


    samples_array = np.concatenate(samples, axis=0)
    orgs_array = np.concatenate(orgs, axis=0)
    labels_array = np.concatenate(labels, axis=0)
    weak_labels_array = np.concatenate(weak_labels, axis=0)
    # Save the samples as a .npz file
    np.savez(args.sample_path,
             samples=samples_array,
             orgs=orgs_array,
             labels=labels_array,
             weak_labels=weak_labels_array)


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
        num_samples=10,
        batch_size=32,   ###1
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=100,
        noise_level=500,
        dataset='brats',
        mask_path="",
        use_resample=False,
        n_generated_batch=10,
        sample_path=""
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()