import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import random
import os
import glob
import timm
import tqdm
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import (DataLoader, TensorDataset, random_split,
                              SubsetRandomSampler, ConcatDataset)

from Utils.Options import Param
from Utils.Dataset import CustomDataset


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class trainer(Param):
    def __init__(self):
        super(trainer, self).__init__()
        os.makedirs(self.OUTPUT_CKP, exist_ok=True)
        os.makedirs(self.OUTPUT_LOSS, exist_ok=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def run(self):
        print('--------------------------------------------------')
        print(f'[DEVICE] : {self.device}')
        print('--------------------------------------------------')

        transform = transforms.Compose(
            [
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
            ]
        )

        fake = glob.glob(f'{os.path.join(self.DATASET_PATH, self.DATA_STYPE[0], self.DATA_CLS[0])}/*')
        real = glob.glob(f'{os.path.join(self.DATASET_PATH, self.DATA_STYPE[0], self.DATA_CLS[1])}/*')

        fake_label = [0] * len(fake)
        real_label = [1] * len(real)

        dataset = fake + real
        label = fake_label + real_label

        dataset_full = CustomDataset(self.DATASET_PATH, self.DATA_STYPE[0], self.DATA_CLS, transform)
        splits = StratifiedKFold(n_splits=self.k, random_state=1004, shuffle=True)

        summary = SummaryWriter(self.OUTPUT_LOSS)
        criterion = nn.CrossEntropyLoss()

        history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
        for fold, (train_idx, val_idx) in enumerate(splits.split(X=dataset, y=label)):
            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(val_idx)
            train_loader = DataLoader(dataset_full, batch_size=self.BATCHSZ, sampler=train_sampler)
            test_loader = DataLoader(dataset_full, batch_size=self.BATCHSZ, sampler=test_sampler)

            model = timm.create_model('swin_small_patch4_window7_224', pretrained=True, num_classes=2)
            model = model.to(self.device)
            optimizer = optim.AdamW(model.parameters(), lr=self.LR)

            for epoch in range(self.MAX_EPOCH):
                train_loss, train_acc, train_f1 = 0, 0, 0
                test_loss, test_acc, test_f1 = 0, 0, 0
                model.train()
                for idx, (item, label) in enumerate(tqdm.tqdm(train_loader, desc=f'Train [{fold+1}-fold]: {epoch}/{self.MAX_EPOCH}')):
                    item = item.to(self.device)
                    label = label.to(self.device)

                    logits = model(item)
                    loss = criterion(logits, label)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    train_acc += (logits.argmax(1) == label).type(torch.float).sum().item()
                    train_f1 += f1_score(label.cpu().numpy(), logits.argmax(1).cpu().numpy(), average='macro')

                with torch.no_grad():
                    model.eval()
                    for idx, (item, label) in enumerate(tqdm.tqdm(test_loader, desc=f'Test [{fold+1}-fold]: {epoch}/{self.MAX_EPOCH}')):
                        item = item.to(self.device)
                        label = label.to(self.device)

                        logits = model(item)
                        loss = criterion(logits, label)

                        test_loss += loss.item()
                        test_acc += (logits.argmax(1) == label).type(torch.float).sum().item()
                        test_f1 += f1_score(label.cpu().numpy(), logits.argmax(1).cpu().numpy(), average='macro')


                train_loss = train_loss / len(train_loader.sampler)
                train_acc = train_acc / len(train_loader.sampler) * 100
                train_f1 = train_f1 / len(train_loader.sampler)

                test_loss = test_loss / len(test_loader.sampler)
                test_acc = test_acc / len(test_loader.sampler) * 100
                test_f1 = test_f1 / len(test_loader.sampler)

                print('\n')
                print('\n')
                print('-------------------------------------------------------------------')
                print(f"Fold[{fold+1}] Epoch: {epoch}/{self.MAX_EPOCH}")
                print(f"Train acc: {train_acc} | Train loss: {train_loss} | Train F1 Score: {train_f1}")
                print(f"Test acc: {test_acc} | Test loss: {test_loss} | Test F1 Score: {test_f1}")
                print('-------------------------------------------------------------------')

                summary.add_scalar(f'{fold + 1}-fold/train/acc', train_acc, epoch)
                summary.add_scalar(f'{fold + 1}-fold/train/loss', train_loss, epoch)
                summary.add_scalar(f'{fold + 1}-fold/train/F1', train_f1, epoch)
                summary.add_scalar(f'{fold + 1}-fold/test/acc', test_acc, epoch)
                summary.add_scalar(f'{fold + 1}-fold/test/loss', test_loss, epoch)
                summary.add_scalar(f'{fold + 1}-fold/test/F1', test_f1, epoch)

                os.makedirs(f'{self.OUTPUT_CKP}/{fold+1}-fold/ckp', exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "AdamW_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                    },
                    os.path.join(f"{self.OUTPUT_CKP}/{fold+1}-fold/ckp", f"{epoch}.pth"),
                )

                history['train_loss'].append(train_loss)
                history['test_loss'].append(test_loss)
                history['train_acc'].append(train_acc)
                history['test_acc'].append(test_acc)

        avg_train_loss = np.mean(history['train_loss'])
        avg_test_loss = np.mean(history['test_loss'])
        avg_train_acc = np.mean(history['train_acc'])
        avg_test_acc = np.mean(history['test_acc'])

        print('\n')
        print('\n')
        print('-------------------------------------------------------------------')
        print(f'Performance of {self.k} fold cross validation')
        print(f"Train acc: {avg_train_acc} | Train loss: {avg_train_loss}")
        print(f"Test acc: {avg_test_acc} | Test loss: {avg_test_loss}")
        print('-------------------------------------------------------------------')