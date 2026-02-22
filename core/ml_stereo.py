import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from core.update import BasicMultiUpdateBlock
from core.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from core.corr import CorrBlock1D, CorrBlock1D_1
from core.utils.utils import coords_grid, updisp8
from core.refinement import *


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

class MLStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        context_dims = args.hidden_dims

        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn=args.context_norm, downsample=args.n_downsample)
        self.update_block_0 = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims, cor_dim=8)
        self.update_block_1 = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims, cor_dim=4)
        self.update_block_2 = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims, cor_dim=2)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])
        self.errorAwareRefinement = DRM()

        if args.shared_backbone:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, 'instance', stride=1),
                nn.Conv2d(128, 256, 3, padding=1))
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1


    def upsample_disp(self, disp, mask):

        N, D, H, W = disp.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(factor * disp, [3,3], padding=1)
        up_disp = up_disp.view(N, D, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, D, factor*H, factor*W)


    def forward(self, image1, image2, iters=12, disp_init=None, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            if self.args.shared_backbone:
                *cnet_list, x = self.cnet(torch.cat((image1, image2), dim=0), dual_inp=True, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0]//2)

            else:
                cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.fnet([image1, image2])

            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]

            # 并非在上下文特征上多次运行GRU的conv层，而是只在开始时运行一次
            # inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

        fmap1, fmap2 = fmap1.float(), fmap2.float()
        corr_fn = CorrBlock1D_1(fmap1, fmap2,  num_levels=self.args.corr_levels)

        coords0, coords1 = self.initialize_flow(net_list[0])

        if disp_init is not None:
            coords1 = coords1 + disp_init

        disp_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()

            if itr < 8:
                r = 8
            elif itr < 16:
                r = 4
            else:
                r = 2

            corr = corr_fn(coords1, r) # index correlation volume
            disp = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):

                if itr < 8:
                    net_list, up_mask, delta_disp = self.update_block_0(net_list, inp_list, corr, disp,
                                                                      iter32=self.args.n_gru_layers == 3,
                                                                      iter16=self.args.n_gru_layers >= 2)
                elif itr < 16:
                    net_list, up_mask, delta_disp = self.update_block_1(net_list, inp_list, corr, disp,
                                                                        iter32=self.args.n_gru_layers == 3,
                                                                        iter16=self.args.n_gru_layers >= 2)
                else:
                    net_list, up_mask, delta_disp = self.update_block_2(net_list, inp_list, corr, disp,
                                                                        iter32=self.args.n_gru_layers == 3,
                                                                        iter16=self.args.n_gru_layers >= 2)


            # in stereo mode, project flow onto epipolar
            delta_disp[:,1] = 0.0

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_disp

            # 我们不需要在test_mode中上采样或输出中间结果
            if test_mode and itr < iters-1:
                continue

            # 上采样视差图
            if up_mask is None:
                disp_up = updisp8(coords1 - coords0)
            else:
                disp_up = self.upsample_disp(coords1 - coords0, up_mask)

            disp_up = disp_up[:,:1]

            if itr == iters - 1:
                if disp_up.max() < 0:
                    refine_value = self.errorAwareRefinement(disp_up, image1, image2)
                    disp_up = disp_up + refine_value
                else:
                    pass

            disp_predictions.append(disp_up)

        if test_mode:
            return coords1 - coords0, disp_up

        return disp_predictions


