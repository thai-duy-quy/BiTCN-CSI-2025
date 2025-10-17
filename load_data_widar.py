import os
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.io import loadmat
import numpy as np 

class WidarDataset(Dataset):
    def __init__(self, env_roots, env=1):
        self.data_struct = []
        self.env = env
        self.env_roots = env_roots
        for file in os.listdir(self.env_roots):
            if file in self.__get_dir_env():
                file_path = self.env_roots + '/'+file
                print(file_path)
                for user in os.listdir(file_path):
                    bvp_file = file_path+'/'+user
                    for file in os.listdir(bvp_file):
                        path_file = bvp_file + '/' + file
                        arr_f = file.split('-')
                        if arr_f[1] in self.__get_class(): 
                            struct = {
                                'path':path_file,
                                'user': arr_f[0].replace('user',''),
                                'class': int(arr_f[1])-1,
                                'loc': arr_f[2],
                                'ori':arr_f[3]
                            }
                            self.data_struct.append(struct)
    def __get_dir_env(self):
        if self.env == '1':
            return ['20181109-VS','20181115-VS','20181130-VS']
    def __get_class(self):
        return ['1','2','3']
    def __len__(self):
        return len(self.data_struct)
    
    def __getitem__(self, idx):
        struct = self.data_struct[idx]
        path = struct['path']
        data = loadmat(path)['velocity_spectrum_ro'].astype(np.float32)
        #print(data.dtype)
        #exit()
        data = torch.from_numpy(data.reshape((1,data.shape[2],400)))
        # print(data.shape)
        # exit()
        label = struct['class']
        return data, label


def pad_collate_fn(batch):
    mags, labels = zip(*batch)
    max_T = max(x.shape[1] for x in mags)
    
    mags_padded = [F.pad(x, (0, 0, 0, max_T - x.shape[1])) for x in mags]  # pad T dimension
    mags_tensor = torch.stack(mags_padded)  # [B, 4, T_max, 2025]
    labels_tensor = torch.tensor(labels)

    return mags_tensor, labels_tensor


def get_dataloaders_widar(env_roots='data/BVP', batch_size=8, num_workers=4, train_split=0.8, shuffle=True, env='1'):
    dataset = WidarDataset(env_roots=env_roots,env=env)

    # 切分 train/val
    total_len = len(dataset)
    train_len = int(total_len * train_split)
    val_len = total_len - train_len
    print(f'total_len:{total_len}, train_len: {train_len}, val_len: {val_len}')
    #exit()
    
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=pad_collate_fn)

    return train_loader, val_loader

# env = '1'
# widar = WidarDataset('data/BVP',env=env)
# for item, label in widar:
#     print(item.shape)
#     print(label)
# print(len(widar))
#     exit()
# train_loader = DataLoader(widar,batch_size=8, shuffle=True, num_workers=10, collate_fn=pad_collate_fn)
# for item, label in train_loader:
#     print(item)

