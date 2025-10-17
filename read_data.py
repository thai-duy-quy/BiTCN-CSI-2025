import pandas as pd 
import os
import torch 
import numpy as np 
def read_data(env_roots,env_list,csv_path):
    segments = []
    env_roots = env_roots
    env_code_to_name = {"E2": "env1", "E4": "env2"}
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    #print(df)

    for _,row in df.iterrows():
        env_code = row['Env'].strip()
        env = env_code_to_name.get(env_code)
        #print(env)
        if env not in env_list:
            continue

        subject = row['Subject'].strip()
        action = row['Action'].strip()
        label = int(action[1:]) - 1  # A1 → 1, A2 → 2, ...
        #print(subject,action,label)

        pt_path = os.path.join(env_roots[env], subject, action, "frames_pca.pt")
        print(pt_path)

        segment_strings = []
        for col_idx in range(4, 29):  # python index 是從 0 起算
            cell = str(row.iloc[col_idx]).strip()
            if '-' in cell:
                segment_strings.append(cell)
        print(segment_strings)

        # mag_pca_len = torch.load(pt_path, map_location="cpu")["mag_pca"].shape[0]
        # print(mag_pca_len)

        for seg_str in segment_strings:
            start, end = map(int, seg_str.split('-'))
            frame_indices = list(range(start, end + 1))
            #print(frame_indices)
            # valid_indices = [i for i in frame_indices]
            # print(valid_indices)
            #exit()
            segments.append((pt_path, frame_indices, label, env, subject, action))
    #print(segments[len(segments)-1])
    print(segments[0])
    print(len(segments))
    return segments
    #exit()
        
def read_data_segment(segments,idx):
    pt_path, frame_indices, label, env, subject, action = segments[idx]
    data = torch.load(pt_path, map_location='cpu')
    #print(data.keys())
    mag_pca = data['mag_pca']
    print(pt_path)
    print(mag_pca)
    exit()
    mag_frames = [mag_pca[i] for i in frame_indices]
    print(mag_frames)
    exit()
    mag_frames_1 = np.array(mag_frames)
    print(len(mag_frames[0][0][0]))
    print(mag_frames_1.shape)
    mag_tensor = torch.cat(mag_frames, dim=1) # [4, total_T, pca_dim]
    print(mag_tensor.shape)
    exit()

    
    return mag_tensor, label



env_roots={"env1": "./env1_frame_pca"}
csv_path="Action_Segment.csv"
env_list = ['env1']
segments = read_data(env_roots,env_list, csv_path)
print(len(segments))
read_data_segment(segments,3)
import random 
a = [torch.tensor([[[random.random() for _ in range(64)] for _ in range(37)] for _ in range(4)]) for _ in range(86)]
a1 = np.array(a)
print(a1.shape)
result = torch.cat(a,dim=1)
print(result.shape)
