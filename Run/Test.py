import os
import pandas as pd
import timm
import tqdm
from sklearn.metrics import f1_score

import torch
import torchvision.transforms as transforms
from Utils.Options import Param
from Utils.Dataset import CustomDataset
from torch.utils.data import (DataLoader, TensorDataset, random_split,
                              SubsetRandomSampler, ConcatDataset)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 2-fold/epoch8
class Tester(Param):
    def __init__(self):
        super(Tester, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def run(self):
        print('--------------------------------------------------')
        print(f'[DEVICE] : {self.device}')
        print('--------------------------------------------------')

        model = timm.create_model('efficientformerv2_l', pretrained=True, num_classes=2)
        model = model.to(self.device)

        transform = transforms.Compose(
            [
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
            ]
        )
        dataset = CustomDataset(self.DATASET_PATH, self.DATA_STYPE[1], self.DATA_CLS, transform)
        test_loader = DataLoader(dataset, batch_size=128)

        for i in range(2, 3):
            for j in range(2, 3):
                print(f'{self.OUTPUT_CKP}/{i}-fold/ckp/{j}.pth')
                checkpoint = torch.load(f'{self.OUTPUT_CKP}/{i}-fold/ckp/{j}.pth', map_location=self.device)
                model.load_state_dict(checkpoint["model_state_dict"])

                pred_list = []
                name_list = []
                with torch.no_grad():
                    model.eval()
                    for idx, (item, id) in enumerate(tqdm.tqdm(test_loader, desc=f'Test')):
                        item = item.to(self.device)

                        logits = model(item)
                        pred = logits.argmax(1)

                        pred_list += pred.tolist()
                        name_list += id

                df = pd.DataFrame({'ImageId': name_list, 'answer': pred_list})

                os.makedirs(f'D:/AI_CONNECT/backup/efficientformerv2_l/pred/{i}-fold', exist_ok=True)
                df.to_csv(f'D:/AI_CONNECT/backup/efficientformerv2_l/pred/{i}-fold/Efficientformer_HighPassFilter_epoch_{j}.csv', index=False)


a = Tester()
a.run()