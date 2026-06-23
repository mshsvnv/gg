import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import warp, get_robust_weight, rgb_to_yuv420, yuv420_to_rgb
from loss import *

FUSE_UV = True


def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)


def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias),
        nn.PReLU(out_channels)
    )


class ResBlock(nn.Module):
    def __init__(self, in_channels, side_channels, bias=True):
        super(ResBlock, self).__init__()
        self.side_channels = side_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(side_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(in_channels)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PReLU(side_channels)
        )
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out[:, -self.side_channels:, :, :] = self.conv2(out[:, -self.side_channels:, :, :].clone())
        out = self.conv3(out)
        out[:, -self.side_channels:, :, :] = self.conv4(out[:, -self.side_channels:, :, :].clone())
        out = self.prelu(x + self.conv5(out))
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        f1_channels = 14 if FUSE_UV else 16
        self.pyramid1 = nn.Sequential(
            convrelu(1, f1_channels, 3, 2, 1),
            convrelu(f1_channels, f1_channels, 3, 1, 1)
        )
        self.pyramid2 = nn.Sequential(
            convrelu(16, 24, 3, 2, 1),
            convrelu(24, 24, 3, 1, 1)
        )
        self.pyramid3 = nn.Sequential(
            convrelu(24, 36, 3, 2, 1),
            convrelu(36, 36, 3, 1, 1)
        )
        self.pyramid4 = nn.Sequential(
            convrelu(36, 48, 3, 2, 1),
            convrelu(48, 48, 3, 1, 1)
        )

    def forward(self, img, img_uv):
        f1 = self.pyramid1(img)
        f2 = self.pyramid2(torch.cat([f1, img_uv], dim=1) if FUSE_UV else f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f1, f2, f3, f4


class Decoder4(nn.Module):
    def __init__(self):
        super(Decoder4, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(48 + 48 + 1, 48),
            ResBlock(48, 16),
            nn.ConvTranspose2d(48, 40, 4, 2, 1, bias=True)
        )

    def forward(self, f0, f1, embt):
        b, c, h, w = f0.shape
        embt = embt.repeat(1, 1, h, w)
        f_in = torch.cat([f0, f1, embt], 1)
        return self.convblock(f_in)


class Decoder3(nn.Module):
    def __init__(self):
        super(Decoder3, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(112, 108),
            ResBlock(108, 16),
            nn.ConvTranspose2d(108, 28, 4, 2, 1, bias=True)
        )

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        return self.convblock(f_in)


class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(76, 72),
            ResBlock(72, 16),
            nn.ConvTranspose2d(72, 20, 4, 2, 1, bias=True)
        )

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        return self.convblock(f_in)


class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        if FUSE_UV:
            self.convblock = nn.Sequential(
                convrelu(52-6, 48),
                ResBlock(48, 16),
                nn.ConvTranspose2d(48, 6, 4, 2, 1, bias=True)
            )
        else:
            self.convblock = nn.Sequential(
                convrelu(52, 48),
                ResBlock(48, 16),
                nn.ConvTranspose2d(48, 6, 4, 2, 1, bias=True)
            )


    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        return self.convblock(f_in)


class Model(nn.Module):
    def __init__(self, local_rank=-1, lr=1e-4):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder4 = Decoder4()
        self.decoder3 = Decoder3()
        self.decoder2 = Decoder2()
        self.decoder1 = Decoder1()
        self.l1_loss = Charbonnier_L1()
        self.tr_loss = Ternary(7)
        self.rb_loss = Charbonnier_Ada()
        self.gc_loss = Geometry(3)

    def _prepare_yuv(self, img0, img1, imgt=None):
        img0_y, img0_uv = rgb_to_yuv420(img0)
        img1_y, img1_uv = rgb_to_yuv420(img1)

        if imgt is None:
            return img0_y, img0_uv, img1_y, img1_uv, None, None

        imgt_y, imgt_uv = rgb_to_yuv420(imgt)
        return img0_y, img0_uv, img1_y, img1_uv, imgt_y, imgt_uv

    def _predict_yuv(self, img0_y, img0_uv, img1_y, img1_uv, imgt_y, imgt_uv, embt, scale_factor=1.0):
        mean_ = torch.cat([img0_y, img1_y], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0_y = img0_y - mean_
        img1_y = img1_y - mean_
        img0_uv_ = img0_uv - 0.5
        img1_uv_ = img1_uv - 0.5

        img0_ = resize(img0_y, scale_factor=scale_factor)
        img1_ = resize(img1_y, scale_factor=scale_factor)

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_, img0_uv_)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_, img1_uv_)

        out4 = self.decoder4(f0_4, f1_4, embt)
        up_flow0_4 = out4[:, 0:2]
        up_flow1_4 = out4[:, 2:4]
        ft_3_ = out4[:, 4:]

        out3 = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4)
        up_flow0_3 = out3[:, 0:2] + 2.0 * resize(up_flow0_4, scale_factor=2.0)
        up_flow1_3 = out3[:, 2:4] + 2.0 * resize(up_flow1_4, scale_factor=2.0)
        ft_2_ = out3[:, 4:]

        out2 = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3)
        up_flow0_2 = out2[:, 0:2] + 2.0 * resize(up_flow0_3, scale_factor=2.0)
        up_flow1_2 = out2[:, 2:4] + 2.0 * resize(up_flow1_3, scale_factor=2.0)
        ft_1_ = out2[:, 4:-2] if FUSE_UV else out2[:, 4:]
        up_res_uv = out2[:, -2:] if FUSE_UV else None
        up_flow0_uv = up_flow0_2
        up_flow1_uv = up_flow1_2

        out1 = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2)
        up_flow0_1 = out1[:, 0:2] + 2.0 * resize(up_flow0_2, scale_factor=2.0)
        up_flow1_1 = out1[:, 2:4] + 2.0 * resize(up_flow1_2, scale_factor=2.0)
        up_mask_1 = torch.sigmoid(out1[:, 4:5])
        up_mask_uv = torch.sigmoid(torch.nn.functional.avg_pool2d(out1[:, 4:5], 2, 2))
        up_res_y = out1[:, 5:6]

        up_flow0_1 = resize(up_flow0_1, scale_factor=(1.0 / scale_factor)) * (1.0 / scale_factor)
        up_flow1_1 = resize(up_flow1_1, scale_factor=(1.0 / scale_factor)) * (1.0 / scale_factor)
        up_mask_1 = resize(up_mask_1, scale_factor=(1.0 / scale_factor))
        up_res_y = resize(up_res_y, scale_factor=(1.0 / scale_factor))

        y_warp0 = warp(img0_y, up_flow0_1)
        y_warp1 = warp(img1_y, up_flow1_1)
        y_pred = up_mask_1 * y_warp0 + (1 - up_mask_1) * y_warp1 + mean_ + up_res_y

        uv_warp0 = warp(img0_uv_, up_flow0_uv)
        uv_warp1 = warp(img1_uv_, up_flow1_uv)
        uv_pred = up_mask_uv * uv_warp0 + (1.0 - up_mask_uv) * uv_warp1 + 0.5
        if FUSE_UV:
            uv_pred = uv_pred + up_res_uv

        decoder_feats = (ft_1_, ft_2_, ft_3_)
        target_flows = (up_flow0_1, up_flow1_1, up_flow0_2, up_flow1_2, up_flow0_3, up_flow1_3, up_flow0_4, up_flow1_4)
        return y_pred, uv_pred, decoder_feats, target_flows

    def inference(self, img0, img1, embt, scale_factor=1.0):
        y_pred, uv_pred, _, _ = self._predict_yuv(*self._prepare_yuv(img0, img1), embt, scale_factor=scale_factor)
        return yuv420_to_rgb(y_pred, uv_pred)

    def inference_yuv(self, img0_y, img0_uv, img1_y, img1_uv, embt, scale_factor=1.0):
        y_pred, uv_pred, _, _ = self._predict_yuv(
            img0_y, img0_uv, img1_y, img1_uv, None, None, embt, scale_factor=scale_factor
        )
        return y_pred, uv_pred

    def forward(self, img0, img1, embt, imgt, flow=None):
        y_pred, uv_pred, decoder_feats, target_flows = self._predict_yuv(
            *self._prepare_yuv(img0, img1), embt, scale_factor=1.0
        )
        imgt_pred = yuv420_to_rgb(y_pred, uv_pred)
        ft_1_, ft_2_, ft_3_ = decoder_feats
        up_flow0_1, up_flow1_1, up_flow0_2, up_flow1_2, up_flow0_3, up_flow1_3, up_flow0_4, up_flow1_4 = target_flows

        _, _, _, _, imgt_y, imgt_uv = self._prepare_yuv(img0, img1, imgt)
        mean_ = torch.cat([img0[:, 0:1], img1[:, 0:1]], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        imgt_y = imgt_y - mean_
        ft_1, ft_2, ft_3, _ = self.encoder(imgt_y, imgt_uv)

        loss_rec = self.l1_loss(imgt_pred - imgt) + self.tr_loss(imgt_pred, imgt)
        loss_geo = 0.01 * (self.gc_loss(ft_1_, ft_1) + self.gc_loss(ft_2_, ft_2) + self.gc_loss(ft_3_, ft_3))

        if flow is not None:
            robust_weight0 = get_robust_weight(up_flow0_1, flow[:, 0:2], beta=0.3)
            robust_weight1 = get_robust_weight(up_flow1_1, flow[:, 2:4], beta=0.3)
            loss_dis = 0.01 * (
                self.rb_loss(2.0 * resize(up_flow0_2, 2.0) - flow[:, 0:2], weight=robust_weight0) +
                self.rb_loss(2.0 * resize(up_flow1_2, 2.0) - flow[:, 2:4], weight=robust_weight1)
            )
            loss_dis += 0.01 * (
                self.rb_loss(4.0 * resize(up_flow0_3, 4.0) - flow[:, 0:2], weight=robust_weight0) +
                self.rb_loss(4.0 * resize(up_flow1_3, 4.0) - flow[:, 2:4], weight=robust_weight1)
            )
            loss_dis += 0.01 * (
                self.rb_loss(8.0 * resize(up_flow0_4, 8.0) - flow[:, 0:2], weight=robust_weight0) +
                self.rb_loss(8.0 * resize(up_flow1_4, 8.0) - flow[:, 2:4], weight=robust_weight1)
            )
        else:
            loss_dis = 0.00 * loss_geo

        return imgt_pred, loss_rec, loss_geo, loss_dis