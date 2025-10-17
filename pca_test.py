import os
import torch
import numpy as np
from tqdm import tqdm
from incremental_pca import IncrementalPCA
import joblib

def run_incremental_pca_from_frames(env_root, save_root, pca_model_path, n_components=256, batch_size=4, max_files=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_list = []

    for subj in os.listdir(env_root):
        for act in os.listdir(os.path.join(env_root, subj)):
            frame_path = os.path.join(env_root, subj, act, "frames.pt")
            if os.path.exists(frame_path):
                file_list.append((frame_path, subj, act))

    if max_files:
        file_list = file_list[:max_files]

    print(f"🔍 Found {len(file_list)} frame files in {env_root}")

    # === Step 1: 建立 Incremental PCA 模型並訓練 ===
    ipca = IncrementalPCA(n_components=n_components, n_features=2025).to(device)

    for i in tqdm(range(0, len(file_list), batch_size), desc="Training PCA"):
        batch = []
        for j in range(i, min(i + batch_size, len(file_list))):
            try:
                path, _, _ = file_list[j]
                mag = torch.load(path, map_location="cpu")['mag']  # [N, 37×8100]
                for frame in mag:
                    reshaped = frame.reshape(37, 4, 2025).permute(1, 0, 2)  # [4, 37, 2025]
                    flat = reshaped.reshape(-1, 2025)

                    # 預處理（NaN, Inf, clip, normalize）
                    flat = torch.nan_to_num(flat, nan=0.0, posinf=1000.0, neginf=-1000.0)
                    flat = torch.clamp(flat, -1000, 1000)
                    mean = flat.mean(dim=0)
                    std = flat.std(dim=0)
                    std[std < 1e-6] = 1e-6
                    normed = (flat - mean) / std

                    batch.append(normed)
            except Exception as e:
                print(f"❌ Error in {path}: {e}")
        if batch:
            all_tensor = torch.cat(batch, dim=0).to(device)
            ipca.partial_fit(all_tensor)
            del all_tensor, batch
            torch.cuda.empty_cache()

    joblib.dump(ipca.cpu(), pca_model_path)
    print(f"✅ PCA model saved to {pca_model_path}")

    # === Step 2: 投影每筆資料並只儲存 mag_pca ===
    for path, subj, act in tqdm(file_list, desc="Projecting and saving"):
        try:
            data = torch.load(path, map_location="cpu")
            mag_all = data['mag']
            mag_proj_list = []

            for frame in mag_all:
                reshaped = frame.reshape(37, 4, 2025).permute(1, 0, 2)  # [4, 37, 2025]
                flat = reshaped.reshape(-1, 2025)
                flat = torch.nan_to_num(flat, nan=0.0, posinf=1000.0, neginf=-1000.0)
                flat = torch.clamp(flat, -1000, 1000)
                mean = flat.mean(dim=0)
                std = flat.std(dim=0)
                std[std < 1e-6] = 1e-6
                normed = (flat - mean) / std

                proj = ipca.transform(normed).reshape(4, 37, n_components)
                mag_proj_list.append(torch.tensor(proj, dtype=torch.float32))

            mag_proj_all = torch.stack(mag_proj_list)  # [N, 4, 37, 256]

            save_dir = os.path.join(save_root, subj, act)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "frames_pca.pt")

            # ✅ 僅儲存 mag_pca（完全不保留 mag, phase 等其他資料）
            torch.save({'mag_pca': mag_proj_all}, save_path)

        except Exception as e:
            print(f"❌ Failed on {path}: {e}")

    print("✅ All done.")

def run_incremental_pca_from_frames_both(env_root, save_root, pca_model_path, n_components=256, batch_size=4, max_files=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_list = []

    for subj in os.listdir(env_root):
        for act in os.listdir(os.path.join(env_root, subj)):
            frame_path = os.path.join(env_root, subj, act, "frames.pt")
            if os.path.exists(frame_path):
                file_list.append((frame_path, subj, act))

    if max_files:
        file_list = file_list[:max_files]

    print(f"🔍 Found {len(file_list)} frame files in {env_root}")

    # === Step 1: 建立 Incremental PCA 模型並訓練 ===
    ipca = IncrementalPCA(n_components=n_components, n_features=2025).to(device)

    for i in tqdm(range(0, len(file_list), batch_size), desc="Training PCA"):
        batch_mag = []
        batch_phase = []
        for j in range(i, min(i + batch_size, len(file_list))):
            try:
                path, _, _ = file_list[j]
                csi = torch.load(path, map_location="cpu")
                mag = csi['mag']  # [N, 37×8100]
                phase = csi['phase']

                for frame in mag:
                    reshaped = frame.reshape(37, 4, 2025).permute(1, 0, 2)  # [4, 37, 2025]
                    flat = reshaped.reshape(-1, 2025)

                    # 預處理（NaN, Inf, clip, normalize）
                    flat = torch.nan_to_num(flat, nan=0.0, posinf=1000.0, neginf=-1000.0)
                    flat = torch.clamp(flat, -1000, 1000)
                    mean = flat.mean(dim=0)
                    std = flat.std(dim=0)
                    std[std < 1e-6] = 1e-6
                    normed = (flat - mean) / std

                    batch_mag.append(normed)
                for frame in phase:
                    reshaped = frame.reshape(37, 4, 2025).permute(1, 0, 2)  # [4, 37, 2025]
                    flat = reshaped.reshape(-1, 2025)

                    # 預處理（NaN, Inf, clip, normalize）
                    flat = torch.nan_to_num(flat, nan=0.0, posinf=1000.0, neginf=-1000.0)
                    flat = torch.clamp(flat, -1000, 1000)
                    mean = flat.mean(dim=0)
                    std = flat.std(dim=0)
                    std[std < 1e-6] = 1e-6
                    normed = (flat - mean) / std

                    batch_phase.append(normed)
            except Exception as e:
                print(f"❌ Error in {path}: {e}")
        if batch_mag:
            all_tensor = torch.cat(batch_mag, dim=0).to(device)
            ipca.partial_fit(all_tensor)
            del all_tensor, batch_mag
            torch.cuda.empty_cache()
        if batch_phase:
            all_tensor = torch.cat(batch_phase, dim=0).to(device)
            ipca.partial_fit(all_tensor)
            del all_tensor, batch_phase
            torch.cuda.empty_cache()

    joblib.dump(ipca.cpu(), pca_model_path)
    print(f"✅ PCA model saved to {pca_model_path}")

    # === Step 2: 投影每筆資料並只儲存 mag_pca ===
    for path, subj, act in tqdm(file_list, desc="Projecting and saving"):
        try:
            data = torch.load(path, map_location="cpu")
            mag_all = data['mag']
            phase_all = data['phase']
            mag_proj_list = []
            phase_proj_list = []

            for frame in mag_all:
                reshaped = frame.reshape(37, 4, 2025).permute(1, 0, 2)  # [4, 37, 2025]
                flat = reshaped.reshape(-1, 2025)
                flat = torch.nan_to_num(flat, nan=0.0, posinf=1000.0, neginf=-1000.0)
                flat = torch.clamp(flat, -1000, 1000)
                mean = flat.mean(dim=0)
                std = flat.std(dim=0)
                std[std < 1e-6] = 1e-6
                normed = (flat - mean) / std

                proj = ipca.transform(normed).reshape(4, 37, n_components)
                mag_proj_list.append(torch.tensor(proj, dtype=torch.float32))

            mag_proj_all = torch.stack(mag_proj_list)  # [N, 4, 37, 256]

            for frame in phase_all:
                reshaped = frame.reshape(37, 4, 2025).permute(1, 0, 2)  # [4, 37, 2025]
                flat = reshaped.reshape(-1, 2025)
                flat = torch.nan_to_num(flat, nan=0.0, posinf=1000.0, neginf=-1000.0)
                flat = torch.clamp(flat, -1000, 1000)
                mean = flat.mean(dim=0)
                std = flat.std(dim=0)
                std[std < 1e-6] = 1e-6
                normed = (flat - mean) / std

                proj = ipca.transform(normed).reshape(4, 37, n_components)
                phase_proj_list.append(torch.tensor(proj, dtype=torch.float32))

            phase_proj_all = torch.stack(phase_proj_list)  # [N, 4, 37, 256]

            save_dir = os.path.join(save_root, subj, act)
            os.makedirs(save_dir, exist_ok=True)
            
            save_path_mag = os.path.join(save_dir, "frames_mag_pca.pt")
            torch.save({'mag_pca': mag_proj_all}, save_path_mag)

            save_path_phase = os.path.join(save_dir, "frames_phase_pca.pt")
            torch.save({'phase_pca': phase_proj_all}, save_path_phase)

        except Exception as e:
            print(f"❌ Failed on {path}: {e}")

    print("✅ All done.")

# =============================================
# 💡 註解：如要為 phase 做相同行為
# → 將 data['mag'] 替換為 data['phase']
# → 將變數名稱改為 phase_proj_list, phase_pca 等
# → 並儲存為 {'phase_pca': phase_proj_all}
# =============================================

if __name__ == "__main__":
    # run_incremental_pca_from_frames(
    #     env_root="./env1_frame",
    #     save_root="./env1_frame_pca",
    #     pca_model_path="./checkpoints/env1_mag_pca.pkl",
    #     n_components=64
    # )
    # run_incremental_pca_from_frames(
    #     env_root="./env2_frame",
    #     save_root="./env2_frame_pca",
    #     pca_model_path="./checkpoints/env2_mag_pca.pkl",
    #     n_components=64
    # )
    run_incremental_pca_from_frames_both(
        env_root="./env2_frame",
        save_root="./env2_frame_pca_both",
        pca_model_path="./checkpoints/env2_mag_pca.pkl",
        n_components=64
    )