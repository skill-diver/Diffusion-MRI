from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
from visdom import Visdom
viz = Visdom(port=8850)
import sys
import cv2
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

    # The sampling process
    logger.log("sampling...")

    samples = []
    orgs = []
    labels = []
    weak_labels = []
    i = 0
    temp_batch = None

    tensor_mask = th.from_numpy(np.ones((args.image_size, args.image_size))).float().to(dist_util.dev())
    
    for img in datal:
        if i >= args.n_generated_batch:
            break

        model_kwargs = {}
        print('img', img[0].shape, img[1])

        brain_mask = np.where(img[0] > 0, 1, 0)
        brain_mask_single_channel = th.from_numpy(np.any(brain_mask, axis=1)).float().to(dist_util.dev())
        brain_mask_multi_channel = brain_mask_single_channel.unsqueeze(1).repeat(1, 4, 1, 1)
        print('brain_mask shape', brain_mask_single_channel.shape)

        if args.dataset == 'brats':
            Labelmask = th.where(img[3] > 0, 1, 0)
            number = img[4][0]
            # if img[2] == 0:
            #     continue
            if all(element == 1 for element in img[2]):
                continue

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
            model_kwargs['gt'] = img[0].to(dist_util.dev())
            model_kwargs['gt_keep_mask'] = tensor_mask * brain_mask_multi_channel

            print('y', model_kwargs["y"])
        sample_fn = (
            diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
        )
        print('samplefn', sample_fn)

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

        temp_batch = img
        samples.append(sample.cpu().numpy())
        orgs.append(org.cpu().numpy())
        labels.append(img[3].cpu().numpy())
        weak_labels.append(img[2].cpu().numpy())
        i += 1

    samples_array = np.concatenate(samples, axis=0)
    orgs_array = np.concatenate(orgs, axis=0)
    labels_array = np.concatenate(labels, axis=0)
    weak_labels_array = np.concatenate(weak_labels, axis=0)

    diff = np.sum(abs(samples_array[0, ...] - orgs_array[0, ...]), axis=0)
    diff_normalized = diff / np.max(diff)
    threshold = 0.4
    segmented = diff_normalized > threshold

    segmented = (segmented * 255).astype(np.uint8)
    contours, _ = cv2.findContours(segmented, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small blobs
    min_area = 50  # Minimum area to be considered a valid blob
    large_blobs = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Create an empty mask to store the filtered blobs
    filtered_mask = np.zeros_like(segmented)

    # Draw the filtered blobs on the mask
    cv2.drawContours(filtered_mask, large_blobs, -1, (255), thickness=cv2.FILLED)

    # Smooth the edges using morphological operations
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, kernel, iterations = 2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations = 2)

    closing = closing / np.max(closing)
    pil_image = Image.fromarray(closing.astype(np.uint8)*1, mode='L')
    pil_image.save(args.mask_path)

    free_form_mask = th.from_numpy(closing).float().to(dist_util.dev())

    model_kwargs = {}
    print('img', temp_batch[0].shape, temp_batch[1])

    brain_mask = np.where(temp_batch[0] > 0, 1, 0)
    brain_mask_single_channel = th.from_numpy(np.any(brain_mask, axis=1)).float().to(dist_util.dev())
    brain_mask_multi_channel = brain_mask_single_channel.unsqueeze(1).repeat(1, 4, 1, 1)
    print('brain_mask shape', brain_mask_single_channel.shape)

    if args.dataset == 'brats':
        Labelmask = th.where(temp_batch[3] > 0, 1, 0)
        number = temp_batch[4][0]

    if args.class_cond:

        classes = th.randint(
            low=1, high=2, size=(args.batch_size,), device=dist_util.dev()
        )

        model_kwargs["y"] = classes
        model_kwargs['gt'] = temp_batch[0].to(dist_util.dev())
        model_kwargs['gt_keep_mask'] = free_form_mask * brain_mask_multi_channel

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
    samples_array = np.concatenate(samples, axis=0)


    # Save the samples as a .npz file
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