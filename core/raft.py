import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder, ContextEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8, upflow_n

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        self.fnet = ContextEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
        self.cnet = ContextEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
        self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1
    
    def initialize_flow_mts(self, img, fraction):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//fraction, W//fraction, device=img.device)
        coords1 = coords_grid(N, H//fraction, W//fraction, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        # print(type(image1))
        # print(type(image2))
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        # print("image 1 shape: {}".format(image1.shape))
        # print("image 2 shape: {}".format(image2.shape))

        image1 = image1.contiguous()
        image2 = image2.contiguous()
        # print("after contiguous image 1 shape: {}".format(image1.shape))
        # print("after contiguous image 2 shape: {}".format(image2.shape))

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            # fmap1, fmap2 = self.fnet([image1, image2])
            x1, x2, x3 = self.fnet([image1, image2])
            
        fmap1_8,  fmap2_8 = x3
        fmap1_8 = fmap1_8.float()
        fmap2_8 = fmap2_8.float()

        fmap1_4,  fmap2_4 = x2
        fmap1_4 = fmap1_4.float()
        fmap2_4 = fmap2_4.float()

        fmap1_2,  fmap2_2 = x1
        fmap1_2 = fmap1_2.float()
        fmap2_2 = fmap2_2.float()

        corr_fn_dict = {
            8: CorrBlock(fmap1_8, fmap2_8, radius=self.args.corr_radius),
            4: CorrBlock(fmap1_4, fmap2_4, radius=self.args.corr_radius),
            2: CorrBlock(fmap1_2, fmap2_2, radius=self.args.corr_radius)
        }
        

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet2, cnet4, cnet8 = self.cnet(image1)
            # print("cnet shape: {}".format(cnet.shape))
            net2, inp2 = torch.split(cnet2, [hdim, cdim], dim=1)
            net4, inp4 = torch.split(cnet4, [hdim, cdim], dim=1)
            net8, inp8 = torch.split(cnet8, [hdim, cdim], dim=1)

            net2 = torch.tanh(net2)
            inp2 = torch.relu(inp2)
            net4 = torch.tanh(net4)
            inp4 = torch.relu(inp4)
            net8 = torch.tanh(net8)
            inp8 = torch.relu(inp8)
            net = {8: net8, 4: net4, 2: net2}
            inp = {8: inp8, 4: inp4, 2: inp2}

        coords0_8, coords1_8 = self.initialize_flow_mts(image1, 8)
        coords0_4, coords1_4 = self.initialize_flow_mts(image1, 4)
        coords0_2, coords1_2 = self.initialize_flow_mts(image1, 2)

        coords0_dict = {8: coords0_8, 4: coords0_4, 2: coords0_2}
        coords1_dict = {8: coords1_8, 4: coords1_4, 2: coords1_2}

        scales = [8, 8, 4, 4, 2, 2]

        flow_predictions = []
        for s in scales:
            print("\nscale: {}".format(s))
            coords1 = coords1_dict[s].detach()
            print("coords1 shape: {}".format(coords1.shape))
            corr = corr_fn_dict[s](coords1) # index correlation volume

            flow = coords1 - coords0_dict[s]
            with autocast(enabled=self.args.mixed_precision):
                net[s], up_mask, delta_flow = self.update_block(net[s], inp[s], corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            # print("\nscale = {}".format(s))
            # print("coords1 shape = {}".format(coords1.shape))
            # print("delta flow shape = {}".format(delta_flow.shape))
            coords1_dict[s] = coords1 + delta_flow

            # upsample predictions
            flow_up = upflow_n(coords1 - coords0_dict[s], s)

            # update current prediction to upper scale
            if s > 2:
                coords1_dict[s//2] = upflow_n(coords1_dict[s], 2)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1_dict[2] - coords0_dict[2], flow_up
            
        return flow_predictions
