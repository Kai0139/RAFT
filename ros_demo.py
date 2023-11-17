import sys
sys.path.append('core')

import rospy
from sensor_msgs.msg import Image, CompressedImage
import cv2
from cv_bridge import CvBridge

import numpy as np
import pathlib

import torch
DEVICE = 'cuda'

import argparse

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class RaftROS(object):
    def __init__(self, args) -> None:
        self.img1 = None
        self.img2 = None
        
        self.args = args
        self.model = torch.nn.DataParallel(RAFT(self.args))
        self.model.load_state_dict(torch.load(args.model))

        self.model = self.model.module
        self.model.to(DEVICE)
        self.model.eval()

        print("# of params: {}".format(count_parameters(self.model)))

        self.bridge = CvBridge()
        self.padder = None
        self.img_sub = rospy.Subscriber("/mvsua_cam/image_raw1", Image, self.img_callback)
        # self.flow_pub = rospy.Publisher("/flow/compressed", CompressedImage, queue_size=1)
        self.flow_pub = rospy.Publisher("/flow", Image, queue_size=1)

    def img_callback(self, img_msg):
        if self.img1 is None:
            self.img1 = self.load_img(img_msg)
            self.padder = InputPadder(self.img1.shape)
            print(self.img1.shape)
            self.img1 = self.padder.pad(self.img1)[0]
            return
        

        self.img2 = self.padder.pad(self.load_img(img_msg))[0]

        flow_low, flow_up = self.model(self.img1, self.img2, iters=15, test_mode=True)
        self.visualize(self.img1, flow_up)

        # RAFT
        self.img1 = self.img2.detach()

    def load_img(self, img_msg):
        cv_img = self.bridge.imgmsg_to_cv2(img_msg).imag
        cv_img = cv2.resize(cv_img, (int(cv_img.shape[0]/2), int(cv_img.shape[1]/2)))
        np_img = np.array(cv_img).astype(np.uint8)
        # print("img size: {}".format(np_img.shape))
        img = torch.from_numpy(np_img).permute(2, 0, 1).float()
        return img[None].to(DEVICE)

    def visualize(self, img, flow):
        # img = img[0].permute(1,2,0).cpu().numpy()
        flow = flow[0].permute(1,2,0).cpu().detach().numpy()
        
        # map flow to rgb image
        flow = flow_viz.flow_to_image(flow)
        print(np.array(flow[:, :, [2,1,0]]).shape)
        # img_flow = np.concatenate([img, flow], axis=0)

        # cv_img = img_flow[:, :, [2,1,0]]/255.0
        # cv_img = flow[:, :, [2,1,0]]

        # img_msg = self.bridge.cv2_to_compressed_imgmsg(cv_img, "png")
        img_msg = self.bridge.cv2_to_imgmsg(np.array(flow[:, :, [2,1,0]]).astype(np.uint8), "rgb8")
        self.flow_pub.publish(img_msg)
    
if __name__ == "__main__":
    rospy.init_node("raft_node")
    model_path = str(pathlib.Path(__file__).parent.joinpath("models", "raft-small.pth"))
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default=model_path)
    parser.add_argument('--small', help='use small model', default=True)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    raft_ros = RaftROS(args)
    rospy.spin()
