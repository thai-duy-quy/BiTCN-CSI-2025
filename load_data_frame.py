import os
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

class CSIDataset(Dataset):
    def __init__(self, env_list, env_roots, csv_path="Action_Segment.csv"):
        self.segments = []
        self.env_roots = env_roots
        env_code_to_name = {"E1": "env1", "E2": "env2", "E3": "env3"}

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        for _, row in df.iterrows():
            env_code = row['Env'].strip()
            env = env_code_to_name.get(env_code)
            if env not in env_list:
                continue

            subject = row['Subject'].strip()
            action = row['Action'].strip()
            # if subject != "S1" or action not in ["A1", "A2"]:
            #     continue
            label = int(action[1:]) - 1  # A1 → 1, A2 → 2, ...

            # pt_path = os.path.join(self.env_roots[env], subject, action, "frames.pt")
            pt_path = os.path.join(self.env_roots[env], subject, action, "frames_pca.pt")
            if not os.path.exists(pt_path):
                continue

            # 從第 5 欄 (index 4) 到第 29 欄 (index 28)，收集 segments
            segment_strings = []
            for col_idx in range(4, 29):  # python index 是從 0 起算
                cell = str(row.iloc[col_idx]).strip()
                if '-' in cell:
                    segment_strings.append(cell)

            mag_pca_len = torch.load(pt_path, map_location="cpu")["mag_pca"].shape[0]

            for seg_str in segment_strings:
                start, end = map(int, seg_str.split('-'))
                frame_indices = list(range(start, end + 1))
                # self.segments.append((pt_path, frame_indices, label, env, subject, action))
                valid_indices = [i for i in frame_indices if i < mag_pca_len]
                if valid_indices:
                    self.segments.append((pt_path, valid_indices, label, env, subject, action))
            
            # print(self.segments)
            # exit()


    def __len__(self):
        return len(self.segments)

    # def __getitem__(self, idx):
    #     pt_path, frame_indices, label, env, subject, action = self.segments[idx]
    #     data = torch.load(pt_path, map_location='cpu')

    #     # mag_frames = [data['mag'][i].reshape(37, 4, 2025).permute(1, 0, 2) for i in frame_indices]
    #     # mag_tensor = torch.cat(mag_frames, dim=1)  # [4, total_T, 2025]
    #     mag_frames = [data['mag_pca'][i] for i in frame_indices]  # 每個 shape: [4, 37, pca_dim]
    #     mag_tensor = torch.cat(mag_frames, dim=1)  # [4, total_T, pca_dim]

    #     return mag_tensor, label

    def __getitem__(self, idx):
        pt_path, frame_indices, label, env, subject, action = self.segments[idx]
        data = torch.load(pt_path, map_location='cpu')
        #print(data)
        
        mag_pca = data['mag_pca']
        #print(mag_pca.max())
        #print(mag_pca.min())
        #print(mag_pca.shape)
        #exit()

        valid_frames = [i for i in frame_indices if i < mag_pca.shape[0]]
        #print(valid_frames)

        if len(valid_frames) == 0:
            raise ValueError(f"❌ All frame_indices out-of-bounds in {pt_path}: requested {frame_indices}, available: {mag_pca.shape[0]}")

        mag_frames = [mag_pca[i] for i in valid_frames]
        mag_tensor = torch.cat(mag_frames, dim=1)  # [4, total_T, pca_dim]
        # print(mag_tensor.shape)
        # exit()

        return mag_tensor, label


def pad_collate_fn(batch):
    mags, labels = zip(*batch)
    max_T = max(x.shape[1] for x in mags)

    mags_padded = [F.pad(x, (0, 0, 0, max_T - x.shape[1])) for x in mags]  # pad T dimension
    mags_tensor = torch.stack(mags_padded)  # [B, 4, T_max, 2025]
    labels_tensor = torch.tensor(labels)

    return mags_tensor, labels_tensor


def get_dataloaders(env_list, env_roots, csv_path="Action_Segment.csv", batch_size=8, num_workers=4, train_split=0.8, shuffle=True):
    dataset = CSIDataset(env_list=env_list, env_roots=env_roots, csv_path=csv_path)

    # 切分 train/val
    total_len = len(dataset)
    train_len = int(total_len * train_split)
    val_len = total_len - train_len
    print(f'total_len:{total_len}, train_len: {train_len}, val_len: {val_len}')
    
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    # print(len(train_set),len(val_set))
    #print(train_set)
    #exit()
    # 分別建立 dataloader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=pad_collate_fn)

    return train_loader, val_loader

