'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import random
import numpy as np

import torch.nn as nn
import torch.nn.init as init
import torch

# coding: utf-8
import csv
'''
写读追加状态
'r'：读
'w'：写
'a'：追加
'r+' == r+w（可读可写，文件若不存在就报错(IOError)）
'w+' == w+r（可读可写，文件若不存在就创建）
'a+' ==a+r（可追加可写，文件若不存在就创建）
对应的，如果是二进制文件，就都加一个b就好啦：
'rb'　　'wb'　　'ab'　　'rb+'　　'wb+'　　'ab+'
'''
# read csv file, return list
def read_csv(file_name):
    with open(file_name, 'r',encoding='utf-8') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data
 
# write csv file, return nothing
def write_csv(file_name, data, mode='w+'):
    with open(file_name, mode, newline="\n",encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)
 
# dict writ into csv file, return nothing
def write_csv_dict(file_name, data, mode='w+'):
    with open(file_name, mode, newline="\n",encoding='utf-8') as f:
        writer = csv.DictWriter(f, data[0].keys())
        # writer.writeheader()
        writer.writerows(data)
# dict read from csv file, return list
def read_csv_dict(file_name,mode='r'):
    with open(file_name, mode, newline="\n",encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    return data
    
# set the seeds of this program
def setup_seed(seed=0, cuda_deterministic=False):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
     else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def adjust_lr(optimizer, epoch, base_lr, epoches):    
    if(epoch >= 0.75*epoches):
        lr = base_lr*0.01
    elif(epoch >= 0.5*epoches):
        lr = base_lr*0.1
    # elif(epoch >= 0.25*epoches):
        # lr = base_lr*0.1
    else:
        lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def make_dir(path):
    tmp_list=os.path.split(path)
    subset=""
    for i in tmp_list:
        subset=os.path.join(subset,i)
        if not os.path.isdir(subset):
            os.makedirs(subset)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)
term_width = 80

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
