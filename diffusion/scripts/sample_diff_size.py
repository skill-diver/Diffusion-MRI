from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
from visdom import Visdom
viz = Visdom(port=8850)
import sys
sys.path.append("..")
sys.path.append(".")
import cv2
import random
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

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    ds = BRATSDataset(args.data_dir, test_flag=True)
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=True)

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path)
    )

    print('loaded classifier')
    p1 = np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()
    p2 = np.array([np.array(p.shape).prod() for p in classifier.parameters()]).sum()
    print('pmodel', p1, 'pclass', p2)

    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

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

    # 根据给定的大小比例字典，大脑掩码和肿瘤标签，判断肿瘤的大小类别
    def judge_tumor_size(size_ratio_dict, brain_mask_single_channel, label):
        # 计算大脑和肿瘤的面积
        brain_area = np.sum(brain_mask_single_channel)
        tumor_area = np.sum(label)
        # 计算肿瘤占大脑的比例
        ratio = tumor_area / brain_area

        # 根据比例判断肿瘤的大小类别
        if ratio <= size_ratio_dict['small']:
            return 'small', brain_area
        elif size_ratio_dict['small'] < ratio <= size_ratio_dict['medium']:
            return 'medium', brain_area
        else:
            return 'large', brain_area

    # 根据目标面积来调整图像中的白色区域的大小
    def adjust_white_area(img, brain_mask_single_channel, target_area, kernel_size=3):
        # 创建一个指定大小的矩阵，用于图像的膨胀和腐蚀操作
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # 计算图像中白色区域的当前面积
        current_area = np.sum(img == 1)
        img = img.astype(np.uint8)
        
        # 如果当前的白色区域面积小于目标面积，进行膨胀操作
        if current_area < target_area:
            while current_area < target_area:
                img = cv2.dilate(img, kernel, iterations=1)
                img = brain_mask_single_channel * img
                current_area = np.sum(img == 1)
        # 如果当前的白色区域面积大于目标面积，进行腐蚀操作
        elif current_area > target_area:
            while current_area > target_area:
                img = cv2.erode(img, kernel, iterations=1)
                current_area = np.sum(img == 1)
        
        # 返回处理后的图像
        return (img > 0).astype(int)

    # 基于给定的大小比例字典和大脑的面积，为其他两种大小类别生成目标面积
    def get_other_two_sizes(size_ratio_dict, other_keys, brain_area):
        target_areas = {}
        for key in other_keys:
            if key == 'small':
                target_areas['small'] = random.uniform(0, size_ratio_dict['small']) * brain_area
            elif key == 'medium':
                target_areas['medium'] = random.uniform(size_ratio_dict['small'], size_ratio_dict['medium']) * brain_area
            else:
                target_areas['large'] = random.uniform(size_ratio_dict['medium'], size_ratio_dict['large']) * brain_area
        return target_areas



    # The sampling process
    logger.log("sampling...")

    samples = []
    orgs = []
    labels = []
    weak_labels = []
    i = 0
    size_ratio_dict = {'small': 0.05, 'medium': 0.2, 'large': 0.3}
    temp_batch = None

    pil_mask = Image.open(args.mask_path)
    tensor_mask = th.from_numpy(np.array(pil_mask)).float().to(dist_util.dev())

    for img in datal:
        if i >= args.n_generated_batch:
            break
        if all(element == 0 for element in img[2]):
            continue
        temp_batch = img
        i += 1

    model_kwargs = {}
    print('img', temp_batch[0].shape, temp_batch[1])

    brain_mask = np.where(temp_batch[0] > 0, 1, 0)
    brain_mask_single_channel = th.from_numpy(np.any(brain_mask, axis=1)).float().to(dist_util.dev())
    brain_mask_multi_channel = brain_mask_single_channel.unsqueeze(1).repeat(1, 4, 1, 1)
    print('brain_mask shape', brain_mask_single_channel.shape)
    brain_mask_single_channel = brain_mask_single_channel.cpu().numpy()

    label_binary = np.where(temp_batch[3] > 0, 1, 0)

    size_key, brain_area = judge_tumor_size(size_ratio_dict, brain_mask_single_channel, label_binary)

    other_keys = [key for key in size_ratio_dict.keys() if key != size_key]

    target_areas = get_other_two_sizes(size_ratio_dict, other_keys, brain_area)

    for key in target_areas.keys():
        target_area = target_areas[key]
        print("target_key", key)
        print("target_area", target_area)
        label_new = adjust_white_area(label_binary[0], brain_mask_single_channel[0], target_area, kernel_size=3)
        tensor_mask = th.from_numpy(np.array(label_new)).float().to(dist_util.dev())

        if args.class_cond:
            # classes = th.randint(
            #     low=0, high=1, size=(args.batch_size,), device=dist_util.dev()
            # )
            classes = th.randint(
                low=1, high=2, size=(args.batch_size,), device=dist_util.dev()
            )

            # Initialization of model_kwargs. This is used for model conditioning.
            # y: target class as weak labels.
            # gt: original image as the repaint pixel-wise conditioning
            # gt_keep_mask: mask indicating where to generate new partition of the image.

            model_kwargs["y"] = classes
            model_kwargs['gt'] = temp_batch[0].to(dist_util.dev())
            model_kwargs['gt_keep_mask'] = tensor_mask * brain_mask_multi_channel

            print('y', model_kwargs["y"])
        sample_fn = (
            diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
        )
        print('samplefn', sample_fn)

        sample, x_noisy, org = sample_fn(
            model_fn,
            (args.batch_size, 4, args.image_size, args.image_size), temp_batch, org=temp_batch,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
            noise_level=args.noise_level,
            use_resample=args.use_resample
        )
        samples.append(sample.cpu().numpy())

        labels.append(model_kwargs['gt_keep_mask'].cpu().numpy()[0, 0, ...][None, ...])

    samples.append(temp_batch[0].cpu().numpy())
    orgs.append(temp_batch[0].cpu().numpy())
    labels.append(temp_batch[3][0].cpu().numpy())
    weak_labels.append(temp_batch[2].cpu().numpy())

    samples_array = np.concatenate(samples, axis=0)
    orgs_array = np.concatenate(orgs, axis=0)
    labels_array = np.concatenate(labels, axis=0)
    weak_labels_array = np.concatenate(weak_labels, axis=0)

    np.savez(args.sample_path,
             samples=samples_array,
             orgs=orgs_array,
             labels=labels_array,
             weak_labels=weak_labels_array)



def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_samples=10,
        batch_size=16,   ###1
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=100,
        noise_level=500,
        dataset='brats',
        mask_path="",
        use_resample=False,
        n_generated_batch=1,
        sample_path=""
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()