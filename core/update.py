import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract

class DispHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(DispHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, cz, cr, cq, *x_list):
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)) + cq)

        h = (1-z) * h + z * q
        return h


class ConvGRU1(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, kernel_size=3):
        super(ConvGRU1, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h



class MAC_GRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, small_kernel_size=1, large_kernel_size=3):
        super(MAC_GRU, self).__init__()
        self.small_gru = ConvGRU1(hidden_dim, input_dim, small_kernel_size)
        self.large_gru = ConvGRU1(hidden_dim, input_dim, large_kernel_size)
        # 用卷积生成像素级 alpha 权重
        self.alpha_generator = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, h, *x):
        x = torch.cat(x, dim=1)
        h_small = self.small_gru(h, x)
        h_large = self.large_gru(h, x)

        combined = torch.cat([h_small, h_large], dim=1)  # [B, 2*hidden_dim, H, W]
        alpha = self.alpha_generator(combined)  # alpha 的形状为 [B, 1, H, W]

        h = h_large * alpha + h_small * (1 - alpha)

        return h



class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, *x):
        # horizontal
        x = torch.cat(x, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, args, cor_planes):
        super(BasicMotionEncoder, self).__init__()
        self.args = args

        # cor_planes = args.corr_levels * (2*args.corr_radius + 1)

        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+64, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)

def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)


class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dims=[], cor_dim=4):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args, cor_planes=args.corr_levels * (2*cor_dim+1))
        encoder_output_dim = 128

        self.gru04 = MAC_GRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1) + hidden_dims[2])
        self.gru08 = MAC_GRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[1] + hidden_dims[2])
        self.gru16 = MAC_GRU(hidden_dims[0], hidden_dims[0] + hidden_dims[1])
        self.flow_head = DispHead(hidden_dims[2], hidden_dim=256, output_dim=2)
        factor = 2**self.args.n_downsample

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, (factor**2)*9, 1, padding=0))


    def forward(self, net, inp, corr=None, disp=None, iter08=True, iter16=True, iter32=True, update=True):

        if iter32:
            net[2] = self.gru16(net[2], (inp[2]), pool2x(net[1]))
        if iter16:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(net[1], (inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], (inp[1]), pool2x(net[0]))
        if iter08:
            motion_features = self.encoder(disp, corr)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru04(net[0], (inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru04(net[0], (inp[0]), motion_features)

        if not update:
            return net

        delta_disp = self.flow_head(net[0])

        # scale mask to balence gradients
        mask = .25 * self.mask(net[0])
        return net, mask, delta_disp
