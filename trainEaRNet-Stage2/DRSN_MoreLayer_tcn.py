import torch
import torch.nn as nn
import math
from ConvtasTCN import *

device = torch.device('cuda')


class SubDifferentialNet3(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(SubDifferentialNet3, self).__init__()
        self.in_channel_conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=mid_channel, kernel_size=3, padding=1))
        self.filter = ResNet(num_res=3, in_channel=mid_channel, out_channel=out_channel)

    # real and imag size = (b, c, m, n)
    def forward(self, real_imag):
        real_imag = self.in_channel_conv1(real_imag)

        real_imag = self.filter(real_imag)
        return real_imag


class ResNet(nn.Module):
    def __init__(self, num_res, in_channel, out_channel):
        super(ResNet, self).__init__()
        self.res_sequential = nn.Sequential()
        for i in range(num_res):
            if i == num_res - 1:
                self.res_sequential.add_module(f'res_conv{(i + 1)}',
                                               BasicBlock(in_channels=in_channel, inner_channel=in_channel // 2,
                                                          out_channels=out_channel))
            else:
                self.res_sequential.add_module(f'res_conv{(i + 1)}',
                                               BasicBlock(in_channels=in_channel, inner_channel=in_channel // 2,
                                                          out_channels=out_channel))
                self.res_sequential.add_module(f'res_relu{(i + 1)}', nn.GELU())  # =========

    def forward(self, src):
        src = self.res_sequential(src)
        return src


class BasicBlock(nn.Module):
    def __init__(self, in_channels, inner_channel, out_channels=64, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, in_channels, stride)
        self.gn1 = nn.GroupNorm(in_channels // 8, in_channels)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.GELU()

        self.conv2 = conv3x3(in_channels, inner_channel, stride)
        self.gn2 = nn.GroupNorm(in_channels // 8, inner_channel)
        # self.bn2 = nn.BatchNorm2d(out_channels)

        self.inner_channel = inner_channel
        self.out_channel = out_channels
        self.conv3 = nn.Conv1d(inner_channel, out_channels, kernel_size=1, stride=1)

        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        if self.inner_channel != self.out_channel:
            out = self.conv3(out)
        out += residual
        return out


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class RSBU_CW(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, down_sample=False):
        super().__init__()
        self.down_sample = down_sample
        self.in_channels = in_channels
        self.out_channels = out_channels
        stride = 1
        if down_sample:
            stride = 2
        self.BRC = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                      padding=1)
        )
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.FC = nn.Sequential(
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.Sigmoid()
        )
        self.flatten = nn.Flatten()
        self.average_pool = nn.AvgPool1d(kernel_size=1, stride=2)

    def forward(self, input):
        x = self.BRC(input)
        x_abs = torch.abs(x)
        gap = self.global_average_pool(x_abs)
        gap = self.flatten(gap)
        alpha = self.FC(gap)
        threshold = torch.mul(gap, alpha)
        threshold = torch.unsqueeze(threshold, 2)
        # 软阈值化
        sub = x_abs - threshold
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x), n_sub)
        if self.down_sample:  # 如果是下采样，则对输入进行平均池化下采样
            input = self.average_pool(input)
        if self.in_channels != self.out_channels:  # 如果输入的通道和输出的通道不一致，则进行padding,直接通过复制拼接矩阵进行padding,原代码是通过填充0
            # 这里是简单乘2的padding，要修改
            xshape = x.size()
            inshape = input.size()
            padshape = (xshape[0],xshape[1]-inshape[1],xshape[2])
            zero_padding = torch.zeros(padshape).cuda()
            input = torch.cat((input, zero_padding), dim=1)

        result = x + input
        return result


class DRSNet(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        # self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.bn = nn.BatchNorm1d(32)
        # self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.rsbu_cw1 = RSBU_CW(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=3,
                                down_sample=False)
        self.rsbu_cw2 = RSBU_CW(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3,
                                down_sample=False)
        self.rsbu_cw3 = RSBU_CW(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3,
                                down_sample=False)

    def forward(self, input):  # 1*256
        x = self.rsbu_cw1(input)
        x = self.rsbu_cw2(x)
        out = self.rsbu_cw3(x)
        return out


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch,kernel_size,stride,padding):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class NestedUNet_DRSN1d_MoreLayer(nn.Module):
    def __init__(self, num_classes, input_channels, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        # self.initNet = nn.Sequential(
        #     nn.Conv1d(input_channels, 32, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm1d(32),
        #     nn.LeakyReLU()
        # )

        self.pool = nn.AvgPool1d(2, 2)

        self.conv0_0 = DRSNet(in_channel=1, out_channel=nb_filter[0])
        self.conv1_0 = DRSNet(in_channel=nb_filter[0], out_channel=nb_filter[1])
        self.conv2_0 = DRSNet(in_channel=nb_filter[1], out_channel=nb_filter[2])
        self.conv3_0 = DRSNet(in_channel=nb_filter[2], out_channel=nb_filter[3])
        self.conv4_0 = DRSNet(in_channel=nb_filter[3], out_channel=nb_filter[4])

        self.tcnNet = TemporalConvNet(in_channel=nb_filter[4], out_channel=nb_filter[4],
                                      TBlock_channel=nb_filter[4] * 2,
                                      kernel_size=3, num_block=3, num_repeat=2)

        self.conv3_1 = SubDifferentialNet3(nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv2_1 = SubDifferentialNet3(nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv2_2 = SubDifferentialNet3(nb_filter[2] * 3, nb_filter[2], nb_filter[2])

        self.conv1_1 = SubDifferentialNet3(nb_filter[1] * 2, nb_filter[1], nb_filter[1])
        self.conv1_2 = SubDifferentialNet3(nb_filter[1] * 3, nb_filter[1], nb_filter[1])
        self.conv1_3 = SubDifferentialNet3(nb_filter[1] * 4, nb_filter[1], nb_filter[1])

        self.conv0_1 = SubDifferentialNet3(nb_filter[0] * 2, nb_filter[0], nb_filter[0])
        self.conv0_2 = SubDifferentialNet3(nb_filter[0] * 3, nb_filter[0], nb_filter[0])
        self.conv0_3 = SubDifferentialNet3(nb_filter[0] * 4, nb_filter[0], nb_filter[0])
        self.conv0_4 = SubDifferentialNet3(nb_filter[0] * 5, nb_filter[0], nb_filter[0])

        self.up1_0 = up_conv(64, 32,4,2,1)
        self.up2_0 = up_conv(128, 64,4,2,1)
        self.up3_0 = up_conv(256, 128,5,2,1)
        self.up4_0 = up_conv(512, 256,4,2,1)

        self.up1_1 = up_conv(64, 32,4,2,1)
        self.up2_1 = up_conv(128, 64,4,2,1)
        self.up3_1 = up_conv(256, 128,5,2,1)

        self.up1_2 = up_conv(64, 32,4,2,1)
        self.up2_2 = up_conv(128, 64,4,2,1)

        self.up1_3 = up_conv(64, 32,4,2,1)

        self.final = nn.Sequential(
            nn.Conv1d(32, num_classes, kernel_size=1)
        )

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x4_0 = self.tcnNet(x4_0)

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_1,x0_0,self.up1_1(x1_1)],1))
        x1_2 = self.conv1_2(torch.cat([x1_0,x1_1,self.up2_1(x2_1)],1))
        x2_2 = self.conv2_2(torch.cat([x2_1,x2_0,self.up3_1(x3_1)],1))

        x0_3 = self.conv0_3(torch.cat([x0_0,x0_1,x0_2,self.up1_2(x1_2)],1))
        x1_3 = self.conv1_3(torch.cat([x1_0,x1_1,x1_2,self.up2_2(x2_2)],1))

        x0_4 = self.conv0_4(torch.cat([x0_0,x0_1,x0_2,x0_3,self.up1_3(x1_3)],1))

        output = self.final(x0_4)

        return output

# if __name__ == "__main__":
#     print("deep_supervision: False")
#     deep_supervision = False
#     device = torch.device('cuda')
#     inputs = torch.randn((4, 1, 900)).to(device)
#     model = NestedUNet_DRSN1d_MoreLayer(num_classes=1,input_channels=1).to(device)
#     outputs = model(inputs)
#     print(outputs.shape)
