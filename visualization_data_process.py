import torch 
import numpy as np 
import matplotlib.pyplot as plt
pt_path ='env1_frame/S1/A1/frames.pt'
pt_path_2 = 'env1_frame_pca/S1/A1/frames_pca.pt'

data = torch.load(pt_path, map_location='cpu')
data_2 = torch.load(pt_path_2, map_location='cpu')
#print(data)
mag = data['phase'].detach().numpy()
mag_2 = data_2['mag_pca'].detach().numpy()

def plot_2D_amp(data,channel):
    for i in range(1700,1705):
        data1 = data[:,i]
        #data1 = data1.mean(axis=1)
        X = np.arange(len(data1))
        plt.plot(X,data1,label=f'subc: {i}')

    plt.title('CSI Phase over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Phase across Subcarriers')
    plt.legend()
    plt.grid(True)
    plt.show()
def plot_3D_amp_1(data, channel):    
    time_steps,  subcarriers = data.shape

    X, Y = np.meshgrid(np.arange(subcarriers), np.arange(time_steps))
    fig = plt.figure(figsize=(12, 9))

    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'CSI Phase, Subcarriers, Time stept on Channel '+str(channel))
    ax.plot_surface(X, Y, data, cmap='viridis')
    ax.set_xlabel('Subcarriers')
    ax.set_ylabel('Time Steps')
    ax.set_zlabel('CSI Phase')
    plt.show()

mag = mag.reshape((-1,4,2025))
channel = 1
data = mag[:35000,channel,:]
print(data[:,1700])
# exit()
#data_2 = mag_2[:35000,channel,:]
plot_2D_amp(data,channel)
#plot_3D_amp_1(data,channel)
mag_2 = mag_2.reshape((-1,4,64))

#plot_3D_amp_1(data_2,channel)
# phase = data['phase'].detach().numpy()
# print(phase)
# mag = mag.reshape((-1,4,2025))
# mag_2 = mag_2.reshape((-1,4,64))
# print(mag.shape)

# data = mag[:35000,:,:]
# data_2 = mag_2[:35000,:,:]

# for i in range(1):
#     data1 = data[:,i,0]
#     #data1 = data1.mean(axis=1)
#     X = np.arange(len(data1))
#     plt.plot(X,data1,color='g')

# plt.title('CSI Amplitude over Time (Mean of 2025 Subcarriers)')
# plt.xlabel('Time Steps')
# plt.ylabel('Mean Amplitude across Subcarriers')
# plt.legend()
# plt.grid(True)
# plt.show()

# for i in range(1):
#     data1 = data_2[:,i,0]
#     #data1 = data1.mean(axis=1)
#     X = np.arange(len(data1))
#     plt.plot(X,data1,color='g')
    
# plt.title('CSI Amplitude over Time (Mean of 2025 Subcarriers)')
# plt.xlabel('Time Steps')
# plt.ylabel('Mean Amplitude across Subcarriers')
# plt.legend()
# plt.grid(True)
# plt.show()

