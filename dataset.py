from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from config import Config as conf

""" 数据读取与分批 
DataLoader: 封装分批读取数据
ImageFolder: 提供数据路径和标签匹配
"""

def load_data(conf, training=True):
    if training:
        data_root = conf.train_root
        transform = conf.train_transform
        batch_size = conf.train_batch_size
    else:
        data_root = conf.test_root
        transform = conf.test_transform
        batch_size = conf.test_batch_size
    data = ImageFolder(data_root, transform=transform)
    class_num = len(data.classes)
    loader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=conf.pin_memory,
        num_workers=conf.num_workers,
        drop_last=True
    )
    return loader, class_num



