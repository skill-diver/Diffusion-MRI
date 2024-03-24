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

from scipy import ndimage

def random_translate_diff(orgs, labels):
    if len(labels.shape) == 3:
      labels = labels[:, np.newaxis, :, :]
    print("Orgs shape:", orgs.shape)  # 打印orgs的形状
    print("Labels shape:", labels.shape)  # 打印labels的形状

    orgs_boundary = np.where(orgs > 0)  # 假设orgs中大于0的是边界
    labels_boundary = np.where(labels > 0)  # 这是假设labels中大于0的值表示边界
    print("labels_boundary:", labels_boundary)
    min_x, max_x = np.min(labels_boundary[3]), np.max(labels_boundary[3])
    min_y, max_y = np.min(labels_boundary[2]), np.max(labels_boundary[2])
   
    diff_height = max_y - min_y + 1
    diff_width = max_x - min_x + 1

    print("Boundary min_x:", min_x, "max_x:", max_x)  # 打印x轴的边界值
    print("Boundary min_y:", min_y, "max_y:", max_y)  # 打印y轴的边界值

    max_shift_left = abs(min_x - np.min(orgs_boundary[3]))
    max_shift_right = abs(np.max(orgs_boundary[3]) - min_x)
    max_shift_up = abs(min_y - np.min(orgs_boundary[2]))
    max_shift_down = abs(np.max(orgs_boundary[2]) - max_y)

    shift_x=0
    shift_y=0
    # 对于x轴的平移
    if max_shift_left > 0 or max_shift_right > 0:
        left_shift = -np.random.randint(0, (3/5)*max_shift_left + 1)
        right_shift = np.random.randint(0, (3/5)*max_shift_right + 1)
        shift_x = np.random.choice([left_shift, right_shift])

    # 对于y轴的平移
    if max_shift_up > 0 or max_shift_down > 0:
        up_shift = -np.random.randint(0, (3/5)*max_shift_up + 1)
        down_shift = np.random.randint(0, (3/5)*max_shift_down + 1)
        shift_y = np.random.choice([up_shift, down_shift])

    print("Shift x:", shift_x)  # 打印x轴的平移距离
    print("Shift y:", shift_y)  # 打印y轴的平移距离

    # 使用scipy.ndimage.shift进行平移
    translated_labels = ndimage.shift(labels, shift=(0, 0, shift_y, shift_x), mode='constant', cval=0)


    new_min_x = max(min_x + shift_x, 0)
    new_max_x = min(max_x + shift_x, orgs.shape[3])
    new_min_y = max(min_y + shift_y, 0)
    new_max_y = min(max_y + shift_y, orgs.shape[2])
    
    background = np.zeros_like(orgs)
    background[0, :, new_min_y:new_max_y, new_min_x:new_max_x] = translated_labels[0, :, new_min_y:new_max_y, new_min_x:new_max_x]

    return background



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

    def adjust_white_area(img, brain_mask_single_channel, target_area, kernel_size=3):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        current_area = np.sum(img == 1)
        img = img.astype(np.uint8)
        print("max_img", np.max(img))
        if current_area < target_area:
            while current_area < target_area:
                img = cv2.dilate(img, kernel, iterations=1)
                img = brain_mask_single_channel * img
                current_area = np.sum(img == 1)

        elif current_area > target_area:
            while current_area > target_area:
                img = cv2.erode(img, kernel, iterations=1)
                current_area = np.sum(img == 1)
        print("max_img", np.max(img))
        return (img > 0).astype(int)

    # The sampling process
    logger.log("sampling...")

    samples = []
    orgs = []
    labels = []
    weak_labels = []
    #labels_new = []
    diffs = []
    i = 0
    batch_idx = 0
    size_ratio_dict = {'small': 0.05, 'medium': 0.2, 'large': 0.3}
    temp_batch = None

    for img in datal:
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
        tumor_area = np.sum(label_binary)
        target_area = tumor_area * 0.5

        # label_new = tumor_area
        # tensor_mask = th.from_numpy(1 - np.array(label_new)).float().to(dist_util.dev())

        if args.class_cond:
            classes = th.randint(
                low=0, high=1, size=(args.batch_size,), device=dist_util.dev()
            )
            # classes = th.randint(
            #     low=1, high=2, size=(args.batch_size,), device=dist_util.dev()
            # )

            # Initialization of model_kwargs. This is used for model conditioning.
            # y: target class as weak labels.
            # gt: original image as the repaint pixel-wise conditioning
            # gt_keep_mask: mask indicating where to generate new partition of the image.

            model_kwargs["y"] = classes
            model_kwargs['gt'] = temp_batch[0].to(dist_util.dev())
            # model_kwargs['gt_keep_mask'] = (1-abs(tensor_mask - th.from_numpy(label_binary[0]).float().to(dist_util.dev()))) * brain_mask_multi_channel
            model_kwargs['gt_keep_mask'] = 1 * brain_mask_multi_channel
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
        #labels_new.append(model_kwargs['gt_keep_mask'].cpu().numpy()[0, 0, ...][None, ...])
        orgs.append(temp_batch[0].cpu().numpy())
        labels.append(temp_batch[3][0].cpu().numpy())
        weak_labels.append(temp_batch[2].cpu().numpy())

        samples_array = np.concatenate(samples, axis=0)
        #labels_new_array = np.concatenate(labels_new, axis=0)
        orgs_array = np.concatenate(orgs, axis=0)
        labels_array = np.concatenate(labels, axis=0)
        weak_labels_array = np.concatenate(weak_labels, axis=0)
       
        #current_diff = labels_array
        print("labels array 0",labels_array[0].shape)
        print("labels array",labels_array.shape)
        #mask_area = np.where(orgs_array > 0)
        translated_diff = random_translate_diff(orgs_array,labels_array)
        diffs.append(translated_diff)
        diffs_array = np.concatenate(diffs, axis=0)
        #diffs_array = np.stack(diffs, axis=0)
        save_path = f"{args.sample_path}_batch_{batch_idx}.npz"
        
        np.savez(save_path,
                samples=samples_array, 
                labels=labels_array,
                orgs=orgs_array,
                diffs=diffs_array,
                weak_labels=weak_labels_array)
        
        # 清除列表以便下一次迭代
        samples.clear()
        #labels_new.clear() 
        orgs.clear()
        labels.clear()
        weak_labels.clear()
        diffs.clear()
        batch_idx += 1



def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_samples=10000,
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
    