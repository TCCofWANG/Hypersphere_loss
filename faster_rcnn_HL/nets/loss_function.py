import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

# roi_score =>logit [128,21]
# gt_roi_label =>label [128,]
#feature =>[64*128,2048]

nCenter = 2*torch.rand(21, 2048)-1
nCenter = nCenter.renorm(2,0,1e-5).mul(1e5)
epsilon = 1e-2

class SoftmaxLoss(nn.Module):
    def __init__(self, feat_dim, class_num):
        super(SoftmaxLoss, self).__init__()
        self.feat_dim = feat_dim
        self.class_num = class_num
        self.score = nn.Linear(2048, class_num, bias=False)
        # self.weight = nn.Parameter(10 * torch.randn(class_num, feat_dim))
        # nn.init.xavier_uniform_(self.weight)
        # nn.init.xavier_uniform_(self.scales)
        self.score.weight.data.normal_(0, 0.01)

    def forward(self, features, labels):
        # nweight = self.weight
        # n = features.size(0)
        # features = features.view(-1,2048)
        if labels != None:
            labels = [val for sublist in labels for val in sublist]
            roi_scores      = self.score(features)
        else:
            roi_scores = self.score(features)
        return roi_scores

class CosineMarginProduct(nn.Module):
    def __init__(self, feat_dim=2048, class_num=21, s=6.0, m=0.2):
        super(CosineMarginProduct, self).__init__()
        self.s = s
        self.m = m
        self.score = nn.Linear(feat_dim, class_num, bias=False)
        self.score.weight.data.normal_(0, 0.01)


    def forward(self, features, labels):
        if labels != None:
            labels = torch.Tensor([val for sublist in labels for val in sublist])
            cosine = F.linear(F.normalize(features, dim=1), F.normalize(self.score.weight.data, dim=0))
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.view(-1, 1), 1.0)
            roi_scores = self.s * (cosine - one_hot * self.m)
        else:
            roi_scores = self.score(features)
        return roi_scores





# class gCL(nn.Module):
#     # generaliezed Center Loss
#     def __init__(self, loss_weight=0.1, radius=40, alpha=40, rho=0.5, n_class=21):
#         # alpha denotes the half of inter-class distance in the ideal state
#         super(gCL, self).__init__()
#         self.radius = radius
#         self.loss = nn.MSELoss()
#         self.loss_weight = loss_weight
#         self.alpha = alpha
#         self.rho = rho
#         self.weight = nn.Parameter(10 * torch.randn(n_class, 2048))
#         self.nCenter = 2*torch.rand(21, 2048)-1
#         self.nCenter.renorm(2,0,1e-5).mul(1e5)
#
#     def get_center(self,feature, gt_roi_label, nCenter, radius=40):
#         print('compute center')
#         class_num = nCenter.shape[0]
#         feat_dim = nCenter.shape[1]
#         Center = radius * nCenter
#
#         if torch.cuda.is_available():
#             Center = Center.cuda()
#             # labelNum = labelNum.to(device)
#             # y1 = y1.to(device)
#             # e = e.to(device)
#
#         with torch.no_grad():
#             C_feat = feature
#             index = torch.arange(len(gt_roi_label))
#
#             for j in range(class_num):
#                 Center[j] += sum(C_feat[index[gt_roi_label == j]])
#
#             nCenter = Center.renorm(2, 0, 1e-5).mul(1e5)
#
#         return nCenter
#
#     def get_center_exclude_outliers(self,feature, gt_roi_label, nCenter, radius=40):
#         print('compute center exclude outliers')
#         class_num = nCenter.shape[0]
#         feat_dim = nCenter.shape[1]
#         Center = radius * nCenter
#         labelNum = torch.zeros(class_num, 1)
#         y1 = torch.zeros(class_num, feat_dim)
#         e = torch.ones(class_num, 1)
#
#         if torch.cuda.is_available():
#             Center = Center.cuda()
#             labelNum = labelNum.cuda()
#             y1 = y1.cuda()
#             e = e.cuda()
#
#         with torch.no_grad():
#             for kk in range(10):
#                 sum_Center = torch.zeros(class_num, feat_dim)
#                 sum_y = torch.zeros(class_num, feat_dim)
#
#                 C_feat = feature
#                 index = torch.arange(len(gt_roi_label))
#
#                 for j in range(class_num):
#                     labelNum[j] += len(index[gt_roi_label == j])
#                     x_tmp = C_feat[index[gt_roi_label == j]] - Center[j]
#                     sum_y[j] += sum(torch.where(abs(x_tmp) < e[j], torch.zeros_like(x_tmp), x_tmp))
#                     y1[j] = sum_y[j] / labelNum[j]
#
#                     sum_Center[j] += sum(C_feat[index[gt_roi_label == j]] - y1[j])
#                     Center[j] = sum_Center[j] / labelNum[j]
#
#                     ## e[j] update method 1:
#                     # sigma = 1.4815*torch.median(abs(y1[j]-torch.median(y1[j])))
#                     # e[j] = min(e[j],3*sigma)
#
#                     ## e[j] update method 2:
#                     Std = torch.std(y1[j])
#                     IQR = abs(torch.quantile(y1[j], 0.75) - torch.quantile(y1[j], 0.25))
#                     sigma = 1.06 * min(Std, IQR / 1.34) * (torch.tensor(len(y1[j])).pow(-0.2))
#                     vec_w = torch.exp(-abs(y1[j]) / (sigma + 1e-5))
#                     e[j] = min(abs(y1[j][vec_w <= epsilon]))
#
#                     ## e[j] update method 3:
#                     # e[j]=torch.quantile(abs(x_tmp),0.75)
#
#                 nCenter = Center.renorm(2, 0, 1e-5).mul(1e5)
#                 Center = radius * nCenter
#                 # print(y1[j])
#                 # print(e[j])
#
#         return nCenter
#
#     def forward(self, feat, label, epoch):
#
#         # logit for softmax
#         nweight = self.weight.renorm(2, 0, 1e-5).mul(1e5)
#         out = torch.matmul(feat, torch.transpose(nweight, 0, 1))
#
#         # renew the center
#         if epoch<=10:
#             self.nCenter = self.get_center(feat, label, self.nCenter, radius=40)
#         else:
#             self.nCenter = self.get_center_exclude_outliers(feat, label, self.nCenter, radius=40)
#
#         # cal the gcl
#         scenter = self.radius * self.nCenter
#         dists = torch.norm(feat - scenter[label], p=2, dim=1, keepdim=False)
#         thres = self.alpha * self.rho
#         indicate = 1 * dists
#         indicate[indicate <= thres] = 0
#         indicate[indicate > thres] = 1
#         gCLloss = self.loss_weight * (torch.sum((indicate * dists) ** 2) / (2 * len(label)))
#         return out, gCLloss
#
#     def predict(self, feat):
#         nweight = self.weight.renorm(2, 0, 1e-5).mul(1e5)
#         out = torch.matmul(feat, torch.transpose(nweight, 0, 1))
#         return out
