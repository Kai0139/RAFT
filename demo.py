import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

import pathlib
import time

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, idx):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.imwrite("results/image{}.png".format(idx), img_flo[:, :, [0,1,2]])
    # cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.ppm')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        itr = 0
        print(images)
        for i in range(len(images)//2):
            image1 = load_image(images[i*2])
            image2 = load_image(images[i*2+1])
        # for imfile1, imfile2 in zip(images[:-1], images[1:]):
        #     image1 = load_image(imfile1)
        #     image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            start = time.time()
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            print("time cost: {}".format(time.time() - start))

            viz(image1, flow_up, itr)
            itr+=1


if __name__ == '__main__':
    model_path = str(pathlib.Path(__file__).parent.joinpath("models", "raft-chairs-small.pth"))
    img_path = str(pathlib.Path(__file__).parent.joinpath("chair_frames"))
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default=model_path)
    parser.add_argument('--path', help="dataset for evaluation", default=img_path)
    parser.add_argument('--small', help='use small model', default=True)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
