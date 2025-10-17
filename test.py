# import os
# import torch
# from tqdm import tqdm

# def clean_frames_pca(root_dir="./env2_frame_pca"):
#     count = 0
#     for subj in os.listdir(root_dir):
#         for act in os.listdir(os.path.join(root_dir, subj)):
#             file_path = os.path.join(root_dir, subj, act, "frames_pca.pt")
#             if not os.path.exists(file_path):
#                 continue

#             try:
#                 data = torch.load(file_path, map_location='cpu')
#                 if 'mag_pca' not in data:
#                     print(f"⚠️  Skipped {file_path}, no 'mag_pca'")
#                     continue

#                 # 保留只有 mag_pca 的資料
#                 clean_data = {'mag_pca': data['mag_pca']}
#                 torch.save(clean_data, file_path)
#                 count += 1
#             except Exception as e:
#                 print(f"❌ Failed to process {file_path}: {e}")

#     print(f"✅ Cleaned {count} files in {root_dir}")

# if __name__ == "__main__":
#     clean_frames_pca("./env2_frame_pca")  # 可替換為你的路徑


import os
import torch
import pandas as pd

def check_segment_bounds(csv_path, env_root_pca, env_code_to_name={"E2": "env1", "E4": "env2"}):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    print(f"\n🔍 Checking segment indices vs frames_pca.pt availability...")

    for _, row in df.iterrows():
        env_code = row['Env'].strip()

        if env_code not in ["E2", "E4"]:
            continue  # ✅ 只處理 E2 和 E4

        env = env_code_to_name.get(env_code)
        subject = row['Subject'].strip()
        action = row['Action'].strip()
        
        pt_path = os.path.join(env_root_pca[env], subject, action, "frames_pca.pt")
        if not os.path.exists(pt_path):
            print(f"❌ Missing file: {pt_path}")
            continue

        try:
            data = torch.load(pt_path, map_location='cpu')
            num_frames = data['mag_pca'].shape[0]
        except Exception as e:
            print(f"❌ Failed to read {pt_path}: {e}")
            continue

        for col_idx in range(4, 29):  # CSV segment cols
            cell = str(row.iloc[col_idx]).strip()
            if '-' not in cell:
                continue
            start, end = map(int, cell.split('-'))
            if end >= num_frames:
                print(f"⚠️ Out-of-bounds segment in {env}/{subject}/{action}: {cell} (max available frame: {num_frames - 1})")

    print("✅ Done checking.")

# 用法
check_segment_bounds(
    csv_path="Action_Segment.csv",
    env_code_to_name={
        "E2": "env1",
        "E4": "env2"
    },
    env_root_pca={
        "env1": "./env1_frame_pca",
        "env2": "./env2_frame_pca"
    }
)
