import os
import numpy as np
import torch

input_root = r"I:\np_CSI_data\np_data_0514"
output_root = r"I:\np_CSI_data\env3_frame"
frame_len = 37

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for s in range(1, 9):  # S1 ~ S8
    for a in range(1, 10):  # A1 ~ A9
        sid = f"S{s}"
        aid = f"A{a}"

        subfolder = os.path.join(input_root, sid, aid)
        mag_path = os.path.join(subfolder, f"{sid}{aid}_Mag.npz")
        phase_path = os.path.join(subfolder, f"{sid}{aid}_Phase.npz")

        if not (os.path.exists(mag_path) and os.path.exists(phase_path)):
            print(f"找不到 {sid}-{aid} 的資料，略過")
            continue

        # 載入資料
        mag_np = np.load(mag_path)['arr_0']
        phase_np = np.load(phase_path)['arr_0']
        assert mag_np.shape == phase_np.shape, f"{sid}-{aid} Mag/Phase shape 不一致"

        # 轉 GPU tensor
        mag = torch.tensor(mag_np, dtype=torch.float32, device=device)
        phase = torch.tensor(phase_np, dtype=torch.float32, device=device)

        # 切 frame
        total_len = mag.shape[0]
        num_frames = (total_len + frame_len - 1) // frame_len

        mag_frames = []
        phase_frames = []

        for i in range(num_frames):
            start = i * frame_len
            end = start + frame_len

            mag_frame = mag[start:end]
            phase_frame = phase[start:end]

            if mag_frame.shape[0] < frame_len:
                pad_rows = frame_len - mag_frame.shape[0]
                mag_frame = torch.cat([mag_frame, torch.zeros((pad_rows, mag.shape[1]), device=device)], dim=0)
                phase_frame = torch.cat([phase_frame, torch.zeros((pad_rows, phase.shape[1]), device=device)], dim=0)

            mag_frames.append(mag_frame)
            phase_frames.append(phase_frame)

        # 合併 frame
        mag_tensor = torch.stack(mag_frames)
        phase_tensor = torch.stack(phase_frames)

        # 儲存
        out_dir = os.path.join(output_root, sid, aid)
        os.makedirs(out_dir, exist_ok=True)
        pt_save_path = os.path.join(out_dir, "frames.pt")
        torch.save({'mag': mag_tensor, 'phase': phase_tensor}, pt_save_path)

        print(f"{sid}/{aid} 已儲存 {pt_save_path}，shape = {mag_tensor.shape}")
        # 清除 GPU 記憶體（釋放 tensor）
        del mag, phase, mag_tensor, phase_tensor, mag_frames, phase_frames
        torch.cuda.empty_cache()

# === 處理 bg 資料 ===
bg_input_dir = r"I:\np_CSI_data\np_data_0514\bg"
bg_output_dir = r"I:\np_CSI_data\env3_frame\bg"
os.makedirs(bg_output_dir, exist_ok=True)

mag_path = os.path.join(bg_input_dir, "Mag.npz")
phase_path = os.path.join(bg_input_dir, "Phase.npz")

if os.path.exists(mag_path) and os.path.exists(phase_path):
    mag_np = np.load(mag_path)['arr_0']
    phase_np = np.load(phase_path)['arr_0']
    assert mag_np.shape == phase_np.shape, "bg 資料 Mag/Phase shape 不一致"

    mag = torch.tensor(mag_np, dtype=torch.float32, device=device)
    phase = torch.tensor(phase_np, dtype=torch.float32, device=device)

    total_len = mag.shape[0]
    num_frames = (total_len + frame_len - 1) // frame_len

    mag_frames = []
    phase_frames = []

    for i in range(num_frames):
        start = i * frame_len
        end = start + frame_len

        mag_frame = mag[start:end]
        phase_frame = phase[start:end]

        if mag_frame.shape[0] < frame_len:
            pad_rows = frame_len - mag_frame.shape[0]
            mag_frame = torch.cat([mag_frame, torch.zeros((pad_rows, mag.shape[1]), device=device)], dim=0)
            phase_frame = torch.cat([phase_frame, torch.zeros((pad_rows, phase.shape[1]), device=device)], dim=0)

        mag_frames.append(mag_frame)
        phase_frames.append(phase_frame)

    mag_tensor = torch.stack(mag_frames)
    phase_tensor = torch.stack(phase_frames)

    pt_save_path = os.path.join(bg_output_dir, "frames.pt")
    torch.save({'mag': mag_tensor, 'phase': phase_tensor}, pt_save_path)

    print(f"bg 已儲存 {pt_save_path}，shape = {mag_tensor.shape}")

    # 釋放 GPU 記憶體
    del mag, phase, mag_tensor, phase_tensor, mag_frames, phase_frames
    torch.cuda.empty_cache()
else:
    print("找不到 bg 資料")