import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

"""
搭建 mobilenet 框架作为face recognition的特征提取模块
mobilenet: google针对移动终端、嵌入式设备提出的轻量级网络模型
           基本单元是深度级可分离卷积（depthwise separable convolution）
           其可以分解为两个更小的操作：depthwise convolution和pointwise convolution
"""

class Flatten(nn.Module):
    """
    展平模块，全连接层之间将tensor转化为二维
    """
    def forward(self, x):
        return x.view(x.shape[0], -1)

class ConvBn(nn.Module):
    """
    卷积-批归一化模块
    groups: 分组卷积 每组参数被重复使用groups次，参数量更小
    """
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=1, padding=0, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x):
        return self.net(x)

class ConvBnPrelu(nn.Module):
    """
    卷积-批归一化-激活层 C-B-R
    """
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=1, padding=0, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBn(in_c, out_c, kernel, stride, padding, groups),
            nn.PReLU(out_c)
        )

    def forward(self, x):
        return self.net(x)

class DepthWise(nn.Module):
    """
    深度卷积模块：一个卷积核只有一维，一个通道只被一个卷积核卷积
    """
    def __init__(self, in_c, out_c, kernel=(3, 3), stride=2, padding=1, groups=1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBnPrelu(in_c, groups, kernel=(1, 1), stride=1, padding=0),
            ConvBnPrelu(groups, groups, kernel=kernel, stride=stride, padding=padding, groups=groups),
            ConvBn(groups, out_c, kernel=(1, 1), stride=1, padding=0)
        )

    def forward(self, x):
        return self.net(x)

class DepthWiseRes(nn.Module):
    """
    深度卷积-残差模块
    """
    def __init__(self, in_c, out_c, kernel=(3, 3), stride=2, padding=1, groups=1):
        super().__init__()
        self.net = DepthWise(in_c, out_c, kernel, stride, padding, groups)

    def forward(self, x):
        return self.net(x) + x

class MultiDepthWiseRes(nn.Module):
    """
    多通道深度卷积-残差：传入参数num_block，决定堆积的深度卷积块
    """
    def __init__(self, num_block, channels, kernel=(3, 3), stride=1, padding=1, groups=1):
        super().__init__()
        self.net = nn.Sequential(*[
            DepthWiseRes(channels, channels, kernel, stride, padding, groups)
            for _ in range(num_block)
        ])

    def forward(self, x):
        return self.net(x)

class FaceMobileNet(nn.Module):
    """
    搭建MobilenetV1结构
    embedding_size：最终提取的表征人脸的向量大小
    """
    def __init__(self, embedding_size):
        super().__init__()   # input: 1*128*128
        self.conv1 = ConvBnPrelu(1, 64, kernel=(3, 3), stride=2, padding=1)  # 64*64*64
        self.conv2 = ConvBn(64, 64, kernel=(3, 3), stride=1, padding=1, groups=64)  # 64*64*64
        self.conv3 = DepthWise(64, 64, kernel=(3, 3), stride=2, padding=1, groups=128)  # 64*32*32
        self.conv4 = MultiDepthWiseRes(num_block=4, channels=64, kernel=3, stride=1, padding=1, groups=128)
        self.conv5 = DepthWise(64, 128, kernel=(3, 3), stride=2, padding=1, groups=256)
        self.conv6 = MultiDepthWiseRes(num_block=6, channels=128, kernel=(3, 3), stride=1, padding=1, groups=256)
        self.conv7 = DepthWise(128, 128, kernel=(3, 3), stride=2, padding=1, groups=512)
        self.conv8 = MultiDepthWiseRes(num_block=2, channels=128, kernel=(3, 3), stride=1, padding=1, groups=256)
        self.conv9 = ConvBnPrelu(128, 512, kernel=(1, 1))
        self.conv10 = ConvBn(512, 512, groups=512, kernel=(7, 7))
        self.flatten = Flatten()
        self.linear = nn.Linear(2048, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        out = self.flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return out

if __name__ == "__main__":
    """
    测试网络结构和输出
    """
    x_test = Image.open("..\sample/test.png")
    x_test = x_test.convert('L')  # 转化为灰度图
    x = x_test.resize((128, 128))
    x = np.asarray(x, dtype=np.float32)
    x = x[None, None, ...]
    x = torch.from_numpy(x)
    net = FaceMobileNet(embedding_size=512)
    net.eval()
    with torch.no_grad():
        out = net(x)
    print(out.shape)
    print(out)
