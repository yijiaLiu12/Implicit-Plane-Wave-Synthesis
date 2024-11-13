import torch.nn as nn
import torch
import math

class SpatioAttention(nn.Module):
    def __init__(self, ngf):
        super(SpatioAttention, self).__init__()
        self.convSelf1 = nn.Conv2d(ngf,ngf,1)
        self.convSelf2 = nn.Conv2d(ngf, ngf, 1)
        self.convSelf3 = nn.Conv2d(ngf, ngf, 1)
        self.convCom = nn.Conv2d(ngf, ngf, 1)
        self.BNSelf1 = nn.BatchNorm2d(ngf)
        self.BNSelf2 = nn.BatchNorm2d(ngf)
        self.BNSelf3 = nn.BatchNorm2d(ngf)
        self.BNCom = nn.BatchNorm2d(ngf)


    def forward(self, input):
        self1 = self.BNSelf1(self.convSelf1(input))
        self2 = self.BNSelf2(self.convSelf2(input))
        self3 = self.BNSelf3(self.convSelf3(input))


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        channels = math.floor(in_planes / ratio)
        # print(in_planes, channels)
        if channels == 0:
            channels = in_planes
        self.fc1   = nn.Conv2d(in_channels=in_planes, out_channels=channels, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(channels, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

    # def forward(self, x):
    #     avg_out = self.avg_pool(x)
    #     max_out = self.max_pool(x)
    #     # out = avg_out + max_out
    #     return avg_out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# class LocalAwareAttention(nn.Module):
#     def __init__(self):
#         super(LocalAwareAttention, self).__init__()
#
#         self.avgpool = nn.functional.avg_pool2d(4)
#         self.upsample = nn.functional.upsample()
#
#     def forward(self, x):

class LocalAwareAttention(nn.Module):
    def __init__(self):
        super(LocalAwareAttention, self).__init__()

        # self.avgpool = nn.functional.avg_pool2d()
        self.upsample = nn.Upsample(scale_factor=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        avg  = nn.functional.avg_pool2d(x, kernel_size=4, stride=2, padding=1)
        # print(x.shape[2],x.shape[3])
        # avg = nn.AdaptiveAvgPool2d(output_size=(x.shape[2],x.shape[3]))
        sam = self.upsample(avg)
        # print(x.shape, avg.shape, sam.shape)
        add = x - sam
        beta = 0.1
        mul = beta*self.relu(add)*x
        # mul = beta*self.sigmoid(add)

        # mul1 = self.sigmoid(add)
        # mul1 = torch.add(mul1, 0.5)
        # print(torch.min(mul1),torch.max(mul1))
        # mul = torch.mul(mul1, x)
        mul = mul + x

        return mul

class GlobalAwareAttention(nn.Module):
    def __init__(self, in_channel, out_channel, s1, s2, ratio=16):
        super(GlobalAwareAttention, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(s1,s2))
        print(s1,s2)
        # self.conv1 = []

        s3 = in_channel // ratio
        if s3 == 0:
            s3 = in_channel
        self.conv2 = nn.Conv2d(in_channel, s3, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(s3, out_channel, 1, bias=False)

    def forward(self, x):
        # self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=(x.shape[2], x.shape[3]))
        # print(x.shape[2],x.shape[3])
        a = self.conv1(x)
        print(a.shape)
        b = self.conv2(a)
        b = self.relu(b)
        c = self.conv3(b)
        c = torch.sigmoid(c)
        print(x.shape, c.shape)
        d = torch.mul(x, c)
        # print(d.shape)
        return d

class PixelAwareAttention(nn.Module):
    def __init__(self, nf):
        super(PixelAwareAttention, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.sig(y)
        # y1 = torch.mul(x, y)

        return y