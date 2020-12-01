'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import *
import errno

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--wd', default=5e-4, type=float, help='learning rate')
parser.add_argument('--beta', default=0.95, type=float, help='learning rate')
parser.add_argument('--epsilon', default=0.01, type=float, help='learning rate')


parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--when', nargs="+", type=int, default=[100, 150],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument('--save-dir', type=str,  default='default',
                    help='path to save the final model')
# parser.add_argument('-save_hist', action='store_true')
parser.add_argument('-save_noise', action='store_true')
parser.add_argument('-noise_per_epoch', type=int, default=1)
parser.add_argument('-epoch', type=int, default=200)

# parser.add_argument('-cood_noise_per_epoch', type=int, default=1000)
parser.add_argument('-save_noise_iter', action='store_true')
parser.add_argument('-opt', type=str,  default='sgd',
                    help='sgd | adapsgd | adam')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

if args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
elif args.opt=="adapsgd":
    optimizer = AdapSGD(net.parameters(), lr=args.lr, beta=args.beta, epsilon=args.epsilon,
                 weight_decay=args.wd)
elif args.opt=="adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)

else:
    raise Exception('opt option not recognized')
    
    
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
  

# Training

def compute_grad_epoch(epoch):
    print('\nEpoch: %d compute full grad' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer.zero_grad()

    datanum = len(trainloader)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
#         optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)/datanum
        loss.backward(retain_graph=True)
#         optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    true_grad = {}
    clone_grad(net, true_grad)
    return true_grad

def compute_sto_grad_norm(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    noise_sq = []
    stograd_sq = []
    
    true_grads = compute_grad_epoch(epoch)
    gradnorm = compute_norm(true_grads) 
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
#         optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        sto_grads = {}
        clone_grad(net, sto_grads)
        instance_noisesq, instance_gradsq = compute_noise(sto_grads, true_grads)
        noise_sq.append(instance_noisesq)
        stograd_sq.append(instance_gradsq)
        
        
    return noise_sq, stograd_sq, gradnorm

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    noise_sq = []
    stograd_sq = []
    
    if args.save_noise_iter:
        noise_file = save_dir + ('ep%d-noise-periter.csv' % epoch)
        grad_file = save_dir + ('ep%d-stograd-periter.csv' % epoch)
        
        with open(noise_file, 'w') as noisef, open(grad_file, 'w') as gradf:
            noisef.write('noise norms\n')
            gradf.write('stograd norms\n')
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        
        if args.save_noise_iter:
            noise_sq, stograd_sq, true_gradnorm = compute_sto_grad_norm(epoch)
            sto_grad_norm = np.mean(stograd_sq)
            sto_noise_norm = np.mean(noise_sq)
            noise_sq.append(sto_noise_norm)
            stograd_sq.append(sto_grad_norm)
            with open(noise_file, 'a') as noisef, open(grad_file, 'a') as gradf:
                noisef.write('%8.5f\n' % sto_noise_norm)
                gradf.write('%8.5f\n' % sto_grad_norm)
                
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        

    return train_loss/(batch_idx+1), 100.*correct/total

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    return test_loss/(batch_idx+1), acc

        
save_dir = './ckpt/' + args.save_dir + '/'
log_train_file = save_dir + 'train.log'
log_valid_file = save_dir + 'valid.log'
try:
    os.makedirs(save_dir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
    log_tf.write('epoch,loss,accu,gradnorm,sto_grad_norm,noisenorm\n')
    log_vf.write('epoch,loss,accu\n')
    
    
for epoch in range(start_epoch, args.epoch):
    true_gradnorm, sto_grad_norm, sto_noise_norm = 0,0,0
    if args.save_noise and (epoch % args.noise_per_epoch == 0):
        noise_sq, stograd_sq, true_gradnorm = compute_sto_grad_norm(epoch)
        sto_grad_norm = np.mean(stograd_sq)
        sto_noise_norm = np.mean(noise_sq)

    train_loss, train_accu = train(epoch)
    val_loss, val_accu = test(epoch)
    
    if epoch in args.when:
        print('Saving model before learning rate decreased')
#         model_save('{}.e{}'.format(args.save, epoch))
        print('Dividing learning rate by 10')
        for g in optimizer.param_groups:
            g['lr'] /= 10        
            

    with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
        log_tf.write('{epoch},{loss: 8.5f},{accu: 8.5f},{gradnorm:3.3f},{sto_grad_norm:3.3f},{noisenorm:3.3f}\n'.format(
            epoch=epoch, loss=train_loss,
            accu=train_accu, 
            gradnorm=true_gradnorm, sto_grad_norm=sto_grad_norm, noisenorm=sto_noise_norm))
        log_vf.write('{epoch},{loss: 8.5f},{val_accu: 8.5f}\n'.format(
            epoch=epoch, loss=val_loss,
            val_accu=val_accu))            

            
            
        