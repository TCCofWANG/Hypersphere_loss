# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 20:33:26 2022

@author: howarh


Hypersphere Loss V1(calculate the center by EWMA every iteration)
It update the center once every mini-batch.
The centers have inter-class separation.
(The class center is a number of small areas.) 
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import datetime

from models import *
from utils import *

# In[]: Hyper parameters
data_name = 'CIFAR10'
loss_name = 'HL'
model_name = 'ResNet18'
base_lr=0.1
Num_iter = 0
Num_center = 1   # 类中心更新次数
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
stard_seed = 6
total_seed = 6
radius = 40
alpha = 0.998*radius
rho = 0.5  #和CCL2中rho的含义不同，这里是指一般化中心半径占其理论最大半径的比例
loss_weight = 0.002   # constrainted_center_loss前的权重
sep_weight = 0.1    # 不同类别center分离乘法项的权重
Center_step = 1   # 每隔多少步进行一次center更新
iter_pre = 195*5    # constrainted_center_loss加入前模型的运行次数
class_num = 10
feat_dim = 512
batch_size = 256
nCenter = 2*torch.rand(class_num, feat_dim)-1
nCenter = torch.zeros(class_num, feat_dim)
# nCenter = nCenter.renorm(2,0,1e-5).mul(1e5)
epoches = 240
cudaID = [0,1]
time_list = []

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--cudaID', default=cudaID, nargs='+', type=int, help='main cuda ID')
parser.add_argument('--lr', default=base_lr, type=float, help='learning rate')
parser.add_argument('--loss_weight', default=loss_weight, type=float, help='weight of the constrainted center loss')
parser.add_argument('--sep_weight', default=sep_weight, type=float, help='weight of the inter-class seperated constraints')
parser.add_argument('--rho', default=rho, type=float, 
                    help='0-1, the size of general center, 1 denotes the general center is the largest')
parser.add_argument('--lr_milestones', type=str, default='120,180,220')
parser.add_argument('--resume', '-r', action='store_true', default=True, help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda:'+str(args.cudaID[0]) if torch.cuda.is_available() else 'cpu'
# print("Using {} device".format(device))

# In[]: update center
# 基于指数加权平均来进行center更新
def get_center_momentum(im, label, Structure, nCenter, Num_center, sep_weight=0.1, radius = 40, beta=0.99):
    
    # print('compute center')
    class_num = nCenter.shape[0]
    feat_dim = nCenter.shape[1]
    # labelNum = torch.zeros(class_num)
    Center = torch.zeros(class_num, feat_dim)
    CenterMean = torch.zeros(class_num, feat_dim)
    
    with torch.no_grad():
        
        if torch.cuda.is_available():
            im = im.to(device)
            label = label.to(device)
            Center = Center.to(device)
            CenterMean = CenterMean.to(device)
            # labelNum = labelNum.to(device)
        
        C_feat = Structure[0](im)
        if Num_center==1:
             for j in range(class_num):
                 Center[j]+=sum(C_feat[label==j])
        else:
            centers=radius*nCenter
            v= beta*centers[label]+(1-beta)*C_feat
            CenterMean = (torch.sum(centers,dim=0)-centers)/(class_num-1)

            for j in range(class_num):
                Center[j]+=torch.sum(v[label==j]-sep_weight*CenterMean[j],dim=0)
        
        nCenter = Center.renorm(2,0,1e-5).mul(1e5)
        
    return nCenter

# In[]: nn_net_loss
class SoftmaxLoss(nn.Module):
    def __init__(self, feat_dim, class_num):
        super(SoftmaxLoss, self).__init__()
        self.feat_dim = feat_dim
        self.class_num = class_num
        self.weight = nn.Parameter(torch.randn(class_num, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, feature, label):
        # nweight = self.weight
        nweight = self.weight.renorm(2,0,1e-5).mul(1e5)
        # nfeature = self.feature.renorm(2,0,1e-5).mul(1e5)
        out = torch.matmul(feature, torch.transpose(nweight, 0, 1).to(device))
        softmaxLoss = F.cross_entropy(out, label) 
        # the cross_entropy here is equal to softmax+NNL (realy cross entropy)
        return out, softmaxLoss

class gCL(nn.Module):
    # generaliezed Center Loss
    def __init__(self, loss_weight=0.1, sep_weight=0.1, radius = 40, alpha = 40, rho=0.5):
        # alpha denotes the half of inter-class distance in the ideal state
        super(gCL, self).__init__()
        self.radius = radius
        self.loss = nn.MSELoss()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.rho = rho
        
    def forward(self, feat, label, center):
        
        scenter = self.radius * center
        dists = torch.norm(feat-scenter[label], p=2, dim=1, keepdim=False)
        
        thres = alpha*rho
        indicate = torch.where(1*dists > thres,1,0)
        indicate1 = torch.where((1*dists > thres) & (1*dists <= 2*thres),1,0)
        
        # 截断式空间域一般类中心
        gCLloss = self.loss_weight*(torch.sum((indicate*dists)**2)/(2*feat.numel()))
            # +sep_weight*torch.sum(CenterMean[label]*normalized_feat)/label.shape[0])
        
        """渐进式类中心参数准备"""
        # kappa = rho/dists
        # kappa[torch.where(kappa.gt(1))]=1
        
        # 渐进式类中心
        # gCLloss = self.loss_weight * (torch.sum((1-kappa)*dists**2)/(2*feat.numel()))
        
        # 渐进式一般类中心
        # gCLloss = self.loss_weight * (torch.sum(((1-kappa)*indicate*dists)**2)/(2*feat.numel()))

        """角度域类中心参数准备"""
        # cos_dists = torch.sum(feat*center[label],dim=1)/torch.norm(feat,dim=1)
        # thres = 0.99
        # indicate = 1*cos_dists
        # indicate[indicate <= thres] = 1
        # indicate[indicate > thres] = 0
        
        # # 截断式角度域一般类中心
        # gCLloss = self.loss_weight * (torch.sum((indicate*dists)**2)/(2*feat.numel()))
        

        return gCLloss


# In[]: Training and Test
# Training
def train(epoch, Structure, Optimizer, epoches, total_step):
    print('Training...')
    print('Epoch: %d' % epoch)
    
    global nCenter,useCenter,Num_iter,Num_center  # Num_center是center更新的次数
    train_loss = 0
    correct = 0
    total = 0
    
    # if torch.cuda.is_available():
        # torch.cuda.synchronize()
    # end = time.time()
    
    #Sets model in training mode
    Structure[0].train()
    Structure[1].train()
    Structure[2].train()
    
    if(epoch >= 0.4*epoches):
        scalar = 1
        sep_weight = scalar*args.sep_weight
    # elif(epoch >= 0.8*epoches):
    #     scalar = 100
    else:
        sep_weight = args.sep_weight
        scalar=1
    sep_weight=torch.linspace(0, args.sep_weight, epoches)[epoch]
    
    for t, (inputs, label) in enumerate(train_loader):
        
        Num_iter += 1
        
        if Num_iter>iter_pre:
            nCenter = get_center_momentum(inputs, label, Structure, nCenter, 
                                          Num_center, sep_weight, radius)
        
        if (Num_iter>iter_pre) and ((Num_iter-1-iter_pre) % Center_step == 0):
            Num_center += 1
            useCenter = nCenter
            
        
        inputs, label = inputs.to(device), label.to(device)
            
        feature = Structure[0](inputs)
        outputs, ceLoss = Structure[1](feature, label)
       
        if Num_iter<=iter_pre:
            loss = ceLoss
        else:
            gCLloss = Structure[2](feature, label, useCenter)
            loss = ceLoss+scalar*gCLloss
        
        Optimizer[0].zero_grad()
        Optimizer[1].zero_grad()
        
        loss.backward()
        
        Optimizer[0].step()
        Optimizer[1].step()
        # sheduler.step()
        
        if (Num_iter>iter_pre) and ((Num_iter-1-iter_pre) % 195 == 0):    
            cos_matrix=torch.zeros(10, 10)
            dist_matrix=torch.zeros(10, 10)
            for i in range(10):
                for j in range(10):
                    cos_matrix[i,j]=torch.sum(nCenter[i]*nCenter[j])
                    dist_matrix[i,j]=torch.norm(radius*nCenter[i]-radius*nCenter[j], p=2, dim=0, keepdim=False)
            print(cos_matrix)
            print(dist_matrix)
            dists = torch.norm(feature-radius*nCenter[label], p=2, dim=1, keepdim=False)
            dists1 = torch.norm(feature-radius*nCenter[0], p=2, dim=1, keepdim=False)
            print(dists[:10])
            print(dists1[:10])
            
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()

    #     progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
    #                   % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if (Num_iter>iter_pre) and (t+1)%int(total_step/3) == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss {:.4f}, ceLoss {:.4f}, centerLoss {:.4f}'
                   .format(epoch, epoches, t+1, total_step, loss.item(), ceLoss.item(), scalar*gCLloss.item()))
            
    acc = 100.*correct/total
    print('Epoch {}, Loss {:.4f}, Accuracy {:.4f}'
           .format(epoch, train_loss/(t+1), acc))

    # if torch.cuda.is_available():
        # torch.cuda.synchronize()
    # time_list.append(time.time() - end)
    return train_loss/(t+1),acc
    
def test(epoch, Structure, epoches, seed):
    print('Testing...')
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    if(epoch >= 0.4*epoches):
        scalar = 1
    # elif(epoch >= 0.8*epoches):
    #     scalar = 100
    else:
        scalar=1
    
    with torch.no_grad():
        for t, (inputs, label) in enumerate(test_loader):
            inputs, label = inputs.to(device), label.to(device)
            
            feature = Structure[0](inputs)
            outputs, ceLoss = Structure[1](feature, label)
            if Num_iter<=iter_pre:
                loss = ceLoss
            else:
                gCLloss = Structure[2](feature, label, useCenter)
                loss = ceLoss+scalar*gCLloss
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

            # progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #               % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print('Test Accuracy by SOFTMAX after {}th epoch: {} %'
              .format(epoch, 100.*correct/total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc and epoch>epoches*0.75:
        print('Saving..')
        state = {
            'net': Structure[0].state_dict(),
            'softmax' :Structure[1].state_dict(),
            'gCL' :Structure[2].state_dict(),
            'nCenter': nCenter,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint/'+loss_name):
            os.mkdir('checkpoint/'+loss_name)
        model_path = './checkpoint/'+loss_name+'/'+model_name+'_seed_{}_ckpt.pth'.format(seed)
        torch.save(state, model_path)
        best_acc = acc
    return test_loss/(t+1),acc

from scipy.io import savemat
def feature_distribution(Structure, test_loader, nCenter):
    
    class_num = nCenter.shape[0]
    feat_dim = nCenter.shape[1]
    results = []
    features = []
    labels = []
    
    with torch.no_grad():
        for t, (inputs, label) in enumerate(test_loader):
            inputs, label = inputs.to(device), label.to(device)
            feature = Structure[0](inputs)
            normalized_feat = feature.renorm(2,0,1e-5).mul(1e5)
            dists = torch.norm(feature-radius*nCenter[label], p=2, dim=1, keepdim=False)
            dists1 = torch.norm(feature-radius*nCenter[1], p=2, dim=1, keepdim=False)
            cos_dists = torch.sum(feature*nCenter[label],dim=1)/torch.norm(feature,dim=1)
            cos_dists1 = torch.sum(feature*nCenter[1],dim=1)/torch.norm(feature,dim=1)
            features.append(normalized_feat.cpu().numpy())
            labels.append(label.cpu().numpy())
            # break
            
        savemat('results/'+data_name+'_'+loss_name+'_'+model_name+'_'+'save.mat',{'features':features,"labels":labels})
        
        results.append(label)
        results.append(dists.to(torch.int32))
        results.append(dists1.to(torch.int32))
        results.append(cos_dists)
        results.append(cos_dists1)    
        torch.set_printoptions(precision=2)
        print(results)
        
        # the similarity between class center and and the class weight vector
        cos_sim = torch.zeros(class_num, class_num)
        for i in range(class_num):
            for j in range(class_num):
                cos_sim[i,j]=torch.sum(Structure[1].weight[i]*nCenter[j])
        print(cos_sim)
            
if __name__ == '__main__':

    for seed in range(stard_seed,total_seed+1):
    
        best_acc = 0  # best test accuracy
        setup_seed(seed)
        
        # In[]:Data
        print('==> Preparing data..')
        transform_train = transforms.Compose([                                   
            # transforms.Resize(64),
            transforms.RandomCrop(32, padding=4), #(32, padding=4)
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),                                  
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

        ])

        transform_test = transforms.Compose([
            # transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='.././data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

        testset = torchvision.datasets.CIFAR10(
            root='.././data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

        for X, y in train_loader:
            print("Shape of X [N, C, H, W]: ", X.shape)
            print("Shape of y: ", y.shape, y.dtype, y[0])
            break
        for X, y in test_loader:
            print("Shape of X [N, C, H, W]: ", X.shape)
            print("Shape of y: ", y.shape, y.dtype, y[0])
            break
        
        # In[]: Model
        print('==> Building model..')
        if model_name == 'VGG19':
            net = VGG('VGG19',feat_dim=feat_dim)
        elif model_name == 'ResNet18':
            net = ResNet18(feat_dim=feat_dim)
        elif model_name == 'ResNet50':
            net = ResNet50(feat_dim=feat_dim)
        elif model_name == 'ResNet101':
            net = ResNet101(feat_dim=feat_dim)
        elif model_name == 'PreActResNet18':
            net = PreActResNet18(feat_dim=feat_dim)
        elif model_name == 'GoogLeNet':
            net = GoogLeNet(feat_dim=feat_dim)
        elif model_name == 'DenseNet121':   
            net = DenseNet121(feat_dim=feat_dim)
        elif model_name == 'ResNeXt29_2x64d':   
            net = ResNeXt29_2x64d(feat_dim=feat_dim)
        elif model_name == 'MobileNet':
            net = MobileNet(feat_dim=feat_dim)
        elif model_name == 'MobileNetV2':
            net = MobileNetV2(feat_dim=feat_dim)
        elif model_name == 'DPN92':
            net = DPN92(feat_dim=feat_dim)
        elif model_name == 'ShuffleNetG2':
            net = ShuffleNetG2(feat_dim=feat_dim)
        elif model_name == 'SENet18':
            net = SENet18(feat_dim=feat_dim)
        elif model_name == 'ShuffleNetV2':
            net = ShuffleNetV2(1,feat_dim=feat_dim)
        elif model_name == 'EfficientNetB0':
            net = EfficientNetB0(feat_dim=feat_dim)
        elif model_name == 'RegNetX_200MF':
            net = RegNetX_200MF(feat_dim=feat_dim)
        elif model_name == 'SimpleDLA':
            net = SimpleDLA(feat_dim=feat_dim)
        elif model_name == 'PyramidNet':
            net = PyramidNet110(feat_dim=feat_dim)

        if device == 'cuda:'+str(args.cudaID[0]):
            net = torch.nn.DataParallel(net,device_ids=args.cudaID)
            cudnn.benchmark = True
        loss_ce = SoftmaxLoss(feat_dim, class_num) #SoftmaxLoss, CenterLoss
        gCL_loss  = gCL(args.loss_weight, args.sep_weight, radius, alpha, args.rho)
        net, loss_ce, gCL_loss, nCenter = net.to(device), loss_ce.to(device), gCL_loss.to(device), nCenter.to(device)
        # print(net)
        
        if args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            model_path = './checkpoint/'+loss_name+'/'+model_name+'_seed_{}_ckpt.pth'.format(seed)
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])
            loss_ce.load_state_dict(checkpoint['softmax'])
            gCL_loss.load_state_dict(checkpoint['gCL'])
            nCenter = checkpoint['nCenter']
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']
            
        optimizer4NN = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        # scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer4NN, T_max=200)
        scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer4NN, milestones=args.lr_milestones, gamma=0.1)
        # scheduler = lr_scheduler.StepLR(optimizer4NN, 20, gamma=0.5)
        optimizer4Loss = torch.optim.SGD(loss_ce.parameters(), lr=args.lr, momentum=0.9)
        # scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer4Loss, T_max=200)
        scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer4Loss, milestones=args.lr_milestones, gamma=0.1)
                
        Structure = [net, loss_ce, gCL_loss]
        Optimizer = [optimizer4NN, optimizer4Loss]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        
        loss_train_list, loss_test_list = [], []
        acc_train_list, acc_test_list = [], []
        for epoch in range(start_epoch, epoches):
            if args.resume:
                loss_test, acc_test = test(epoch, Structure, epoches, seed)
                feature_distribution(Structure, test_loader, nCenter)
                break
            total_step = len(train_loader)
            loss_train, acc_train = train(epoch, Structure, Optimizer, epoches, total_step)
            loss_test, acc_test = test(epoch, Structure, epoches, seed)
            # scheduler1.step()
            # scheduler2.step()
            adjust_lr(Optimizer[0], epoch, args.lr, epoches)
            adjust_lr(Optimizer[1], epoch, args.lr, epoches)
            Structure[2].loss_weight=torch.linspace(0, args.loss_weight, epoches)[epoch]
            print("loss_weight:{}".format(Structure[2].loss_weight))
            
            print('epoch:{}, lr4nn:{}, lr4loss:{}'
            .format(epoch, Optimizer[0].param_groups[0]['lr'], Optimizer[1].param_groups[0]['lr']))
            
            loss_train_list.append(loss_train)
            acc_train_list.append(acc_train)
            loss_test_list.append(loss_test)
            acc_test_list.append(acc_test)
        else:
            loss_train_list = np.array(loss_train_list)
            acc_train_list = np.array(acc_train_list)
            loss_test_list = np.array(loss_test_list)
            acc_test_list = np.array(acc_test_list)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            time_list.append(time.time() - end)
            
            if not os.path.isdir('results/'):
                os.mkdir('results/')
            model_path = './results/lossAndAcc_'+loss_name+'_'+model_name+'_seed_{}.npz'.format(seed)
            np.savez(model_path,loss_train_list,acc_train_list,loss_test_list,acc_test_list,time_list)
            
            log_path = './results/experimental_logs.csv'
            if not os.path.exists(log_path):
                table_head = [['dataset','model','algo','time','LR','lr_scheduler','rho',
                'loss_weight','sep_weight','epoches','batch_size','feat_dim','seed','best_acc']]
                write_csv(log_path, table_head, 'w+')
                
            date_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')# 获取当前系统时间
            a_log = [{'dataset': data_name, 'model': model_name, 'algo': loss_name, 'time': date_time,
                  'LR': args.lr, 'lr_scheduler': 'adjust_lr','rho':args.rho,'loss_weight':args.loss_weight,
                  'sep_weight':args.sep_weight,'epoches':epoches, 'batch_size': batch_size,'feat_dim': feat_dim,
                  'seed': seed, 'best_acc': best_acc}]
            write_csv_dict(log_path, a_log, 'a+')
            continue
