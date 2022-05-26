from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import model.resnet as resnet
from model.MPNCOV import MPNCOV

# SC$^2$AM
class scam(nn.Module):
    def __init__(self, planes):
        super(scam, self).__init__()
        
        self.expansion = 1

        DR_stride = 2
        self.relu = nn.ReLU(inplace=True)
        self.ch_dim = 128
        self.conv_for_DR = nn.Conv2d(
                 planes * self.expansion, self.ch_dim, 
                 kernel_size=1,stride=DR_stride, bias=True)
        self.bn_for_DR = nn.BatchNorm2d(self.ch_dim)
        self.row_bn = nn.BatchNorm2d(self.ch_dim)
        # row-wise conv is realized by group conv
        self.row_conv_group = nn.Conv2d(
                self.ch_dim, 4*self.ch_dim, 
                kernel_size=(self.ch_dim, 1), 
                groups = self.ch_dim, bias=True)
        self.fc_adapt_channels = nn.Conv2d(
                4*self.ch_dim, planes * self.expansion, 
                kernel_size=1, groups=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # NxCxHxW
        # 降维
        out = self.conv_for_DR(x)
        out = self.bn_for_DR(out)
        out = self.relu(out)

        # 计算协方差矩阵
        out = MPNCOV.CovpoolLayer(out) # Nxdxd
        out = out.view(out.size(0), out.size(1), out.size(2), 1).contiguous() # Nxdxdx1

        # 映射为通道注意力权重
        out = self.row_bn(out)
        out = self.row_conv_group(out) # Nx512x1x1
        out = self.fc_adapt_channels(out) #NxCx1x1
        out = self.sigmoid(out) #NxCx1x1
        
        return out

def mask_Generation(feature, alpha):
    batch_size = feature.size(0)
    kernel = feature.size(2)
    sum = torch.sum(feature.detach(), dim=1)

    avg = torch.sum(torch.sum(sum, dim=1), dim=1) / kernel ** 2

    mask = torch.where(sum > alpha * avg.view(batch_size, 1, 1), torch.ones(sum.size()).cuda(),
                       (torch.zeros(sum.size()) + 0.1).cuda())

    mask = mask.unsqueeze(1)
    return mask

# 学习通道重要性
class LCI(nn.Module):
    def __init__(self, planes):
        super(LCI, self).__init__()
        # fc+relu
        self.fc = nn.Linear(planes, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # channel
        batch_size = x.size(0)
        att = self.fc(x)
        att = self.relu(att) + 2 # max(att, 1) 保证(att-1)>0 
        att = att.view(batch_size, 1, 1, 1)

        return att


# ATP
class SNet(nn.Module):
    def __init__(self, planes):
        super(SNet, self).__init__()
        self.conv = nn.Conv2d(planes, 1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)

        return self.softmax(x)


class AlphaNet(nn.Module):
    def __init__(self, planes):
        super(AlphaNet, self).__init__()
        self.avgpool =  nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(planes, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.avgpool(x).view(batch_size, -1)
        x = self.fc(x)
        x = self.relu(x) + 1

        return x


class Net(nn.Module):
    def __init__(self, model_path):
        super(Net, self).__init__()
        # sc$^2$am
        self.ch_dim = 2048
        self.pro_dim = 8192
        
        self.bn00 = nn.BatchNorm2d(self.ch_dim)
        self.bn10 = nn.BatchNorm2d(self.ch_dim)
        self.bn20 = nn.BatchNorm2d(self.ch_dim)
        
        self.sacm0 = scam(planes=self.ch_dim)
        self.sacm1 = scam(planes=self.ch_dim)
        self.sacm2 = scam(planes=self.ch_dim)

        self.lci0 = LCI(planes=self.ch_dim)
        self.lci1 = LCI(planes=self.ch_dim)
        self.lci2 = LCI(planes=self.ch_dim)

        self.proj0 = nn.Conv2d(self.ch_dim, self.pro_dim, kernel_size=1, stride=1)
        self.proj1 = nn.Conv2d(self.ch_dim, self.pro_dim, kernel_size=1, stride=1)
        self.proj2 = nn.Conv2d(self.ch_dim, self.pro_dim, kernel_size=1, stride=1)
        self.bn0 = nn.BatchNorm2d(self.ch_dim)
        self.bn1 = nn.BatchNorm2d(self.ch_dim)
        self.bn2 = nn.BatchNorm2d(self.ch_dim)

        # snet and alphanet
        self.snet0 = SNet(planes=self.pro_dim)
        self.snet2 = SNet(planes=self.pro_dim)
        self.alphanet = AlphaNet(planes=self.pro_dim)

        # fc layer
        # dataset: cub, classes: 200
        self.fc_concat = torch.nn.Linear(8192, 200)

        # 参数初始化
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.Linear):
        #         m.weight.data.normal_(0, 0.01)
        #         m.bias.data.zero_()

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # base model
        self.features = resnet.resnet50(pretrained=True, model_root=model_path)

    def forward(self, x):
        batch_size = x.size(0)
        # 特征提取
        feature4_0, feature4_1, feature4_2 = self.features(x)

        # SC$^2$AM
        scam_weight0 = self.sacm0(feature4_0)
        scam_weight1 = self.sacm1(feature4_1)
        scam_weight2 = self.sacm2(feature4_2)
        
        feature4_0 = self.bn00(feature4_0 * scam_weight0)
        feature4_1 = self.bn00(feature4_1 * scam_weight1)
        feature4_2 = self.bn00(feature4_2 * scam_weight2)

        # AFM
        scam_weight0 = scam_weight0.view(batch_size, -1)
        scam_weight1 = scam_weight1.view(batch_size, -1)
        scam_weight2 = scam_weight2.view(batch_size, -1)
        
        att0 = self.lci0(scam_weight0)
        att1 = self.lci1(scam_weight1)
        att2 = self.lci2(scam_weight2)

        D = (1./3)*(att0 * feature4_0 + att1 * feature4_1 + att2 * feature4_2)
        mask = mask_Generation(D, alpha=0.6)

        feature4_0 = self.bn0(mask * feature4_0)
        feature4_1 = self.bn1(mask * feature4_1)
        feature4_2 = self.bn2(mask * feature4_2)

        # 特征映射
        feature4_0 = self.proj0(feature4_0)
        feature4_1 = self.proj1(feature4_1)
        feature4_2 = self.proj2(feature4_2)

        # snet0
        attsnet0 = self.snet0(feature4_0)   # Wx
        # snet2
        attsnet2 = self.snet2(feature4_2)   # Wz
        # alphanet
        alpha = self.alphanet(feature4_1).view(batch_size, 1, 1, 1) # alpha

        # ATP
        result = torch.sign(feature4_0)*torch.pow(abs(feature4_0), alpha) * \
                    attsnet0*feature4_1*attsnet2*torch.sign(feature4_2)*torch.pow(abs(feature4_2), alpha)
        # GAP
        result = self.avgpool(result).view(batch_size, -1)
        # 公式（10），归一化层
        result = torch.nn.functional.normalize(torch.sign(result) * torch.sqrt(torch.abs(result) + 1e-10))
        
        # 分类层，输出概率分布
        result = self.fc_concat(result)
        return result

