import os
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import FaceMobileNet, ResIRSE
import model.metric as metric
from model.loss import FocalLoss
from dataset import load_data
from config import Config as conf

from argparse import ArgumentParser

""" 命令行超参数设置 """
parser = ArgumentParser(description='Process some integers.')
parser.add_argument('--backbone', default=conf.backbone, type=str, help='this is the backbone of training')
parser.add_argument('--metric', default=conf.metric, type=str, help='this is the metric of training')
parser.add_argument('--loss', default=conf.loss, type=str, help='this is the loss of training')
parser.add_argument('--batch_size', default=conf.train_batch_size, type=int, help='this is the batch size of training samples')
parser.add_argument('--epoch', default=conf.epoch, type=int, help='this is the epoch of training')
parser.add_argument('--optimizer', default=conf.optimizer, type=str, help='this is the optimizer of training')
parser.add_argument('--lr', default=conf.lr, type=float, help='this is the optimizer of training')

args = parser.parse_args()


""" 数据设置 """
dataloader, class_num = load_data(conf, training=True)
embedding_size = conf.embedding_size
device = conf.device

""" 模型设置 """
if args.backbone == 'fmobile':
    net = FaceMobileNet(embedding_size).to(device)
else:
    net = ResIRSE(embedding_size, 0.5).to(device)

if args.metric == 'arcface':
    metric = metric.ArcFace(embedding_size, class_num).to(device)
else:
    metric = metric.CosFace(embedding_size, class_num).to(device)

#net = nn.DataParallel(net)  # GPU并行
# metric = nn.DataParallel(metric)

if args.loss == 'focal_loss':
    Loss = FocalLoss(gamma=2)
else:
    Loss = nn.CrossEntropyLoss()

if args.optimizer == 'sgd':
    optimizer = optim.SGD(
        [{'params': net.parameters()},
        {'params': metric.parameters()}],
        lr=args.lr,
        weight_decay=conf.weight_decay
    )
else:
    optimizer = optim.Adam(
        [{'params': net.parameters()},
         {'params': metric.parameters()}],
        lr=args.lr,
        weight_decay=conf.weight_decay
    )
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=conf.lr_step, gamma=0.1)

""" 权重文件夹 """
os.makedirs(conf.checkpoints, exist_ok=True)

""" Training """
def train():
    net.train()
    interval = 1000
    i = 0
    for epoch in range(args.epoch):
        for data, labels in tqdm(dataloader, desc=f"Epoch {epoch}/{args.epoch}", ascii=True, total=len(dataloader)):
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            embeddings = net(data)
            thetas = metric(embeddings, labels)
            loss = Loss(thetas, labels)
            loss.backward()
            optimizer.step()
            i += 1
            if i % interval == 0:
                print(f'loss:{loss}')
        print(f'Epoch{epoch}/{args.epoch}, Average Loss:{loss}')

        backbone_path = osp.join(args.checkpoints, f"{args.backbone}-{args.metric}-epoch{epoch}.pth")
        torch.save(net.state_dict(), backbone_path)
        scheduler.step()


if __name__ == '__main__':
    train()
