from torchvision import datasets
from torchvision import transforms
import argparse
from torch.utils.data import DataLoader
from models.wresnet import WideResNet
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
# from AGC.AGC import AGC
# from AGC.NoiseAGC import NoiseAGC
import numpy as np
from PIL import Image
from torch.autograd import grad
import math
from torch.utils.tensorboard import SummaryWriter
from os import path
import os
import random 
# from NAGC import NAGC


random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
cudnn.benchmark = True


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if len(target.shape) > 1:
        target = torch.argmax(target, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res


class RandomNoise(object):
    def __init__(self, probability=1):
        self.probability = probability
        self.std = 0.25

    def __call__(self, img):
         
        if np.random.choice([True, False], size=1, p=[self.probability, 1-self.probability]):
            noise = np.random.normal(loc=0, scale=1, size=np.shape(img))
            img = np.array(img).astype(float)/255
            img = 255*np.clip(img + noise * self.std, 0, 1)
            return Image.fromarray(img.astype(np.uint8))
        return img

def hinged_noiseanneal(data):
    """
    data : list of img
    """
    data_len = len(data)
    std = 0.25
    
    for i in range(data_len):
        noise = np.random.normal(loc=0, scale=1, size=np.shape(data[i]))
        
        img = np.array(data[i]).astype(float)/255
        img = 255 * np.clip(img + noise * std, 0, 1)
        data[i] = img
    return data
    
    
def main(args):
    #rn = RandomNoise()
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

    train_dataset = datasets.CIFAR10('./', train=True, transform=transform, download=True)

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    test_dataset = datasets.CIFAR10('./', train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    net_depth, net_width = int(args.select_model[4:6]), int(args.select_model[-1])

    Teacher_model = WideResNet(depth=net_depth, num_classes=10, widen_factor=net_width, dropRate=0.3)
    
    Teacher_model.to(args.device)
    optimizer = torch.optim.SGD(Teacher_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    
    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
    
    if not path.exists(args.logdir):
        os.mkdir(args.logdir)

    writer = SummaryWriter(log_dir=args.logdir)

    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    iter_cnt = 0
    for epoch in range(args.total_epochs):
        
        for iter_, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.to(args.device), labels.to(args.device)
            outputs, *acts = Teacher_model(images)
            classification_loss = criterion(outputs, labels)

            optimizer.zero_grad()
            classification_loss.backward()
            optimizer.step()
            
            iter_cnt += 1
            if iter_ % 100 == 0:
                print("Epoch: {}/{}, Iteration: {}/{}, Loss: {:02.5f}".format(epoch, args.total_epochs, iter_,
                                                                              train_loader.__len__(),
                                                                              classification_loss.item()))

        with torch.no_grad():
            Teacher_model.eval()
            cumulated_acc = 0
            for x, y in train_loader:
                x, y = x.to(args.device), y.to(args.device)
                logits, *activations = Teacher_model(x)
                acc = accuracy(logits.data, y, topk=(1,))[0]
                cumulated_acc += acc
            print("Train Accuracy is {:02.2f} %".format(cumulated_acc / train_loader.__len__() * 100))

        writer.add_scalar(tag="vanilla train_acc", scalar_value=acc, global_step=epoch+1)

        with torch.no_grad():
            Teacher_model.eval()
            cumulated_acc = 0
            for x, y in test_loader:
                x, y = x.to(args.device), y.to(args.device)
                logits, *activations = Teacher_model(x)
                acc = accuracy(logits.data, y, topk=(1,))[0]
                cumulated_acc += acc
            print("Test Accuracy is {:02.2f} %".format(cumulated_acc / test_loader.__len__() * 100))
            Teacher_model.train()
            if best_acc <= cumulated_acc / test_loader.__len__() * 100:
                best_acc = cumulated_acc / test_loader.__len__() * 100
                torch.save(Teacher_model.state_dict(),
                           './Pretrained/CIFAR10/WRN-{}-{}/Teacher_best.ckpt'.format(net_depth, net_width))

        writer.add_scalar(tag="vanilla test_acc", scalar_value=acc, global_step=epoch + 1)

        optim_scheduler.step()
    writer.close()
    print(best_acc)

if __name__ == "__main__":
    num_repeat = 1
    lr_groups = [1e-1]
    batch_groups = [128]
    for _ in range(num_repeat):
        for i in range(len(batch_groups)):
            for j in range(len(lr_groups)):
                parser = argparse.ArgumentParser()
                parser.add_argument('--weight_path', type=str, default='')
                parser.add_argument('--total_epochs', type=int, default=160)
                parser.add_argument('--lr', type=float, default=lr_groups[j])
                parser.add_argument('--batch_size', type=int, default=batch_groups[i], help='total training batch size')
                parser.add_argument('--select_model', type=str, default='WRN-40-2', help='What do you want to train?')
                parser.add_argument('--num_workers', type=int, default=8)
                parser.add_argument('--device', type=str, default="cuda:0")
                parser.add_argument('--logdir', type=str, default="./Last_logs")
                args = parser.parse_args()
                main(args)
