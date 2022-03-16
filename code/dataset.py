import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

def train_transform(x):
    x_t = x/ np.sqrt(np.sum(x**2))
    return x_t

def valid_transform(x):
    x_t = x/ np.sqrt(np.sum(x**2))
    return x_t

def onehot(x,class_num=4):
    out = np.eye(class_num,dtype=np.long)[x]
    return out

class train_dataset(Dataset):
    def __init__(self, df, target, transform = None, transform_y = None) -> None:
        super(train_dataset,self).__init__()
        self.x = df.iloc[:,1:-1]
        self.y = df[target]
        self.transform = transform
        self.transform_y = transform_y

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        X = self.x.iloc[idx]
        Y = self.y.iloc[idx]
        if self.transform_y:
            Y = self.transform_y(Y)

        if self.transform:
            X = self.transform(X)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y)