import argparse
import pandas as pd
import numpy as np
import random
import os
from tqdm import tqdm
from importlib import import_module

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from loss import load_criterion


def seed_everything(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)  # if use multi-GPU
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(seed)
	random.seed(seed)


def train_model(args):
    # 재현성을 위한 seed 고정
    seed_everything(args.seed)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.model_dir + f'/{args.name}', exist_ok=True)
    os.makedirs(args.model_dir + f'/{args.name}/state_dicts', exist_ok=True)

    # GPU 사용
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # model 선언
    model = getattr(import_module('model'),args.model)        #defalt = LSTM_BaseModel
    model = model(device)
    model.to(device)

    # parameter 선언
    epoch_num = args.epochs
    criterion = load_criterion(args.criterion)
    opt_module = getattr(import_module("torch.optim"), args.optimizer)
    optimizer = opt_module(model.parameters(), lr=args.lr)


    # data_set, data_loader
    df = pd.read_csv(args.data_dir)
    train_df, valid_df = train_test_split(df, test_size=args.val_ratio, stratify=df['target'], random_state=args.seed)

    dataset_module = import_module('dataset')
    train_dataset = getattr(dataset_module,args.dataset)(train_df,'target',transform=getattr(dataset_module,args.train_transform))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_dataset = getattr(dataset_module,args.dataset)(valid_df,'target',transform=getattr(dataset_module,args.valid_transform))
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)


    ### 학습 코드 시작
    best_test_accuracy = 0.
    best_test_loss = 9999.

    dataloaders = {
            "train" : train_dataloader,
            "valid" : valid_dataloader
        }

    for epoch in range(1,epoch_num+1):
        for phase in ["train", "valid"]:
            running_loss = 0.
            running_acc = 0.
            if phase == "train":
                model.train()
            elif phase == "valid":
                model.eval()

            for images, labels in tqdm(dataloaders[phase]):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    preds = model(images)
                    preds = preds.view(-1,4)
                    loss = criterion(preds, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item()                  # 한 Batch에서의 loss 값 저장
                    preds_num = torch.argmax(preds,dim=1)
                    running_acc += torch.sum(preds_num == labels) # 한 Batch에서의 Accuracy 값 저장

            # 한 epoch이 모두 종료되었을 때,
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_acc / len(dataloaders[phase].dataset)
            
            print(f"epoch-{epoch} {phase}-데이터 셋 평균 Loss : {epoch_loss:.3f}, 평균 Accuracy : {epoch_acc:.3f}")
            if phase == "valid" and best_test_accuracy < epoch_acc:
                best_test_accuracy = epoch_acc
                torch.save(model.state_dict(), f'./model/{args.name}/state_dicts/{epoch:03d}_{best_test_accuracy:0.4f}.pt')
            if phase == "valid" and best_test_loss > epoch_loss:
                best_test_loss = epoch_loss
    print("학습 종료!")
    print(f"최고 accuracy : {best_test_accuracy}, 최고 낮은 loss : {best_test_loss}")
    torch.save(model, f'./model/{args.name}/model.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #	from dotenv import load_dotenv
    import os
    #load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='train_dataset', help='dataset augmentation type (default: train_dataset)')
    parser.add_argument('--train_transform', type=str, default='train_transform', help='data augmentation type (default: train_transform)')
    parser.add_argument('--valid_transform', type=str, default='valid_transform', help='data augmentation type (default: valid_transform)')
    # parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    # parser.add_argument("--resize", nargs="+", type=list, default=[400, 110], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='LSTM_BaseModel', help='model type (default: LSTM_BaseModel)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    # parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    # parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '../data/train.csv'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()

    train_model(args)