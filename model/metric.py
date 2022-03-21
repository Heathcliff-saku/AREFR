import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
构建人脸之间相似性的度量函数
    传统方式为softmax loss (DeepFace中所使用的)
    改进之后的Additional Margin Metric Loss
    主要有：CosFace(余弦距离) ArcFace(角度距离)
"""

class CosFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)  # 参数初始化

    def forward(self, input, label):
        """
        1、将输入和权重L2规范化，计算夹角cosine
        2. cosin-m 得到phi
        3、将output中正确标签的概率值进行替换(强化概率)
        4、返回放大后的值
        """
        cosin = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosin - self.m
        output = cosin*1.0
        batch_size = len(output)
        output[range(batch_size), label] = phi[range(batch_size), label]
        return output * self.s

class ArcFace(nn.Module):
    def __init__(self, embedding_size, class_num, s=30.0, m=0.50):
        super().__init__()
        self.in_features = embedding_size
        self.out_features = class_num
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(class_num, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi-m)
        self.mm = math.sin(math.pi-m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = ((1-cosine.pow(2)).clamp(0, 1)).sqrt()
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine-self.mm)
        output = cosine * 1.0
        batch_size = len(output)
        output[range(batch_size), label] = phi[range(batch_size), label]
        return output * self.s






