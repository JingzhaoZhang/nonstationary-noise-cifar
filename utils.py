'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch
import torch.optim

class AdapSGD(torch.optim.Optimizer):


    def __init__(self, params, lr=0.1, beta=0.95, epsilon=0.01,
                 weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
#         if momentum < 0.0:
#             raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, beta=0.95, epsilon=0.01,
                        weight_decay=weight_decay)

        super(AdapSGD, self).__init__(params, defaults)
        self.params = []
        for group in self.param_groups:
            for p in group['params']:
                self.params.append(p)
        self.mhatsq = 0
        self.beta = beta
        self.epsilon = epsilon
        


    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        total_norm = torch.clone(torch.norm(torch.stack([torch.norm(p.grad.detach()) 
                                                         for p in self.params 
                                                         if p.grad is not None]))).detach()
        if self.mhatsq == 0:
            self.mhatsq = total_norm**2
        
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
#             momentum = group['momentum']
#             dampening = group['dampening']
#             nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                step = (-group['lr']/(self.mhatsq**0.5 + self.epsilon)).detach()
                p.add_(d_p, alpha=step)
        
        self.mhatsq = self.beta * self.mhatsq + (1 - self.beta) * total_norm**2
        return loss





def compute_noise(stoc_grads, true_grads):
    total_noise_sq = 0 
    total_grad_sq = 0
    for k in stoc_grads.keys():
        total_noise_sq += (stoc_grads[k]- true_grads[k]).norm(2).item() ** 2
        total_grad_sq += stoc_grads[k].norm(2).item() ** 2
    return total_noise_sq, total_grad_sq

def compute_norm(grads):
    total_grad_sq = 0
    for k in grads.keys():
        total_grad_sq += grads[k].norm(2).item() ** 2
    return total_grad_sq

def clone_grad(net, true_grads):
    for name, param in net.named_parameters():
        if param.grad is None:
                    continue
        true_grads[name] = torch.clone(param.grad.data).detach()
        

def coord_noise(stoc_grads, true_grads):
    coordnoise = []
    for k in stoc_grads.keys():
        coordnoise.extend((stoc_grads[k]-
                           true_grads[k]).cpu().numpy().flatten().tolist()[:])
    coordnoise = np.array(coordnoise)
    return coordnoise


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


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

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
