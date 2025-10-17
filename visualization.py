import torch
import matplotlib.pyplot as plt
import pandas as pd

# ==== Step 1: 載入資料 ====
#root = r"I:\np_CSI_data\env1_frame\S1\A1\frames.pt"
root = 'env1_frame_pca/S1/A1/frames_pca.pt'

data = torch.load(root)
frames = data["mag_pca"]  # 或 "phase"

print(frames.shape)
exit()

# ==== Step 2: 載入 CSV，取出前 3 個 segment ====
csv = pd.read_csv("Action_Segment.csv")
row = csv.loc[(csv['Env'] == 'E1') & (csv['Subject'] == 'S1') & (csv['Action'] == 'A1')]

# 查看欄位名稱
print("欄位名稱：", csv.columns.tolist())

# 自動取第 1~3 個 Segment 欄位（第 4~6 欄）
segment_columns = csv.columns[4:7]
segments = [tuple(map(int, row[col].values[0].split('-'))) for col in segment_columns]

# ==== Step 3: 處理每個 segment ====
results = []
for i, (start, end) in enumerate(segments):
    selected = frames[start:end+1]  # shape: (num_frames, 37, 8100)
    flattened = selected.reshape(-1, selected.shape[-1])  # (T, 8100)

    # 補 NaN/Inf 為 0
    invalid_mask = ~torch.isfinite(flattened)
    if invalid_mask.any():
        flattened[invalid_mask] = 0

    # Clip 範圍 [-1000, 1000]
    flattened = torch.clamp(flattened, min=-1000, max=1000)

    # 時間軸的平均 amplitude
    mean_over_time = flattened.mean(dim=1)
    results.append(mean_over_time.cpu())

# ==== Step 4: 畫出三段比較 ====
plt.figure(figsize=(16, 5))
for i, mean_series in enumerate(results):
    plt.plot(mean_series.numpy(), label=f"Segment {i+1} ({segments[i][0]}-{segments[i][1]})")

plt.title("Mean Amplitude Over Time for Segments 1-3 (Env1/S1/A1)")
plt.xlabel("Time Index (frame × timestamp)")
plt.ylabel("Mean Amplitude (clipped ±1000)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
