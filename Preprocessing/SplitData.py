from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os


path = 'D:/AI_CONNECT/DB/train'
cls_list = ['fake_images', 'real_images']
max_fold_len = [2000, 8000]

prior_len = 0
for i in range(len(max_fold_len)):
    fold_data = []
    for cls in cls_list:
        names = os.listdir(f'{path}/{cls}')
        temp = []
        fold_cls_data = names[prior_len:max_fold_len[i]]

        for img in fold_cls_data:
            if cls == 'fake_images':
                temp.append([img, 0])
            else:
                temp.append([img, 1])

        fold_data += temp

    prior_len = max_fold_len[i]

    df = pd.DataFrame(fold_data, columns=['image', 'label'])
    os.makedirs('D:/AI_CONNECT/DB/data_info/', exist_ok=True)
    df.to_csv(f'D:/AI_CONNECT/DB/data_info/{i+1}_fold.csv', index=False)