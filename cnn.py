# utils
from __future__ import print_function
import argparse
from time import time
# torch related
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
# our code
from PitchEstimationDataSet import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before saving the trained model')
parser.add_argument('--save-dir', type=str, default='./', metavar='N',
                    help='save directory of trained models')
parser.add_argument('--save-prefix', type=str, default='model_conv2_onefile', metavar='N',
                    help='prefix of trained models')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
annotations_train="../MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/onefiletest/"
annotations_test="../MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/onefiletest/"

training_set = PitchEstimationDataSet(annotations_train, '../data/onefiletest/')
# print (training_set[150]['image'].shape, training_set[150]['frequency'])
train_loader = DataLoader(training_set,
    batch_size = args.batch_size, shuffle = True, **kwargs)
test_loader = DataLoader(
    PitchEstimationDataSet(annotations_test, '../data/onefiletest', transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # conv: 3*256*256   -->  64*256*256
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3)
        # max pool to 64*128*128
        # conv: 64*128*128  -->  64*128*128
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_drop = nn.Dropout2d()
        # max pool to 64*64*64
        # conv: 64*64*64  -->  96*64*64
        self.conv3 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        # max pool to 96*32*32
        # conv: 96*32*32  -->  64*32*32
        self.conv4 = nn.Conv2d(96, 64, kernel_size=3, padding=1)
        # conv: 64*32*32  -->  32*32*32
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        # max pool to 32*16*16
        # reshape(flatten) to 8192
        self.fc1 = nn.Linear(8192, 2048)
        self.fc2 = nn.Linear(2048, 109)

    def forward(self, x):
        # print('in forward:')
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # print(x.data.shape)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(self.conv4(x))
        x = F.relu(F.max_pool2d(self.conv5(x), 2))
        # print(x.data.shape)
        x = x.view(-1, 8192)
        # print(x.data.shape)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    batch_start = time()
    for batch_idx, dictionary in enumerate(train_loader):
        data = dictionary['image']
        target = dictionary['frequency']
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data).type(torch.FloatTensor), Variable(target).type(torch.LongTensor)
        # print('data & target:')
        # print(data.data.shape)
        # print(target.data.shape)
        optimizer.zero_grad()
        output = model(data)
        # print('output:')
        # print(output.data.shape)
        # print(output.data.max())
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # training log
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime per batch: {:.6f}s'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0],
                (time()-batch_start)/args.log_interval))
            batch_start = time()
        # save trained model
        if batch_idx % args.save_interval == 0:
            save_name = args.save_prefix + '_' + str(batch_idx) + '.pt'
            print('Saving model: ' + save_name)
            # torch.save(model.state_dict(), args.save_dir+save_name)
            torch.save(model, args.save_dir+save_name)

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True).type(torch.FloatTensor), Variable(target).type(torch.LongTensor)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
