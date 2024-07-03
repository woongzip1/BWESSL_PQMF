import os
import shutil
import argparse
import torch
import numpy as np
from scipy.signal import stft
import scipy.signal as sig
import torch.nn.functional as F

from matplotlib import pyplot as plt
import librosa

""" 주어진 디렉토리에서 지정된 확장자를 가진 모든 오디오 파일의 절대 경로를 반환합니다. """
def get_audio_paths(paths: list, file_extensions=['.wav', '.flac']):
    audio_paths = []
    if isinstance(paths, str):
        paths = [paths]
        
    for path in paths:
        for root, dirs, files in os.walk(path):
            audio_paths += [os.path.join(root, file) for file in files if os.path.splitext(file)[-1].lower() in file_extensions]
                        
    audio_paths.sort(key=lambda x: os.path.split(x)[-1])
    
    return audio_paths

def count_audio_files(paths: list, file_extensions=['.wav', '.mp3', '.flac']):
    """ 주어진 디렉토리에서 지정된 확장자를 가진 모든 오디오 파일의 개수를 반환합니다. """
    audio_files = get_audio_paths(paths, file_extensions)
    return len(audio_files)

def check_dir_exist(path_list):
    if type(path_list) == str:
        path_list = [path_list]
        
    for path in path_list:
        if type(path) == str and os.path.splitext(path)[-1] == '' and not os.path.exists(path):
            os.makedirs(path)       

def get_filename(path):
    return os.path.splitext(os.path.basename(path))  

"""input LR path -> LR/train & LR/test""" 
def path_into_traintest(lr_folder_path):
    # LR 폴더 내의 모든 하위 폴더(화자 폴더)를 리스트로 가져옵니다.
    speaker_folders = sorted([f for f in os.listdir(lr_folder_path) if os.path.isdir(os.path.join(lr_folder_path, f))])
    
    # 마지막 9명의 화자를 테스트 세트로 설정합니다.
    test_speakers = speaker_folders[-9:]
    train_speakers = speaker_folders[:-9]

    # Train과 Test 폴더를 생성합니다.
    train_path = os.path.join(lr_folder_path, 'train')
    test_path = os.path.join(lr_folder_path, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # 트레이닝 폴더에 화자 폴더를 이동합니다.
    for speaker in train_speakers:
        original_path = os.path.join(lr_folder_path, speaker)
        destination_path = os.path.join(train_path, speaker)
        shutil.move(original_path, destination_path)

    # 테스트 폴더에 화자 폴더를 이동합니다.
    for speaker in test_speakers:
        original_path = os.path.join(lr_folder_path, speaker)
        destination_path = os.path.join(test_path, speaker)
        shutil.move(original_path, destination_path)

    print(f"Train set and test set have been created at {train_path} and {test_path}.")

def si_sdr(reference_signal, estimated_signal):
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    
    Parameters:
    - reference_signal (torch.Tensor): The reference signal (N, L)
    - estimated_signal (torch.Tensor): The estimated signal (N, L)
    
    Returns:
    - si_sdr (torch.Tensor): The SI-SDR value for each signal in the batch (N,)
    """
    # Ensure the inputs are of the same shape
    assert reference_signal.shape == estimated_signal.shape, "Input and reference signals must have the same shape"
    
    # Check if Shape is -> N x L
    if reference_signal.dim() == 3:
        reference_signal = reference_signal.squeeze(1)
        estimated_signal = estimated_signal.squeeze(1)
    
    # Compute the scaling factor
    reference_signal = reference_signal - reference_signal.mean(dim=1, keepdim=True)
    estimated_signal = estimated_signal - estimated_signal.mean(dim=1, keepdim=True)
    
    s_target = (torch.sum(reference_signal * estimated_signal, dim=1, keepdim=True) / torch.sum(reference_signal ** 2, dim=1, keepdim=True)) * reference_signal
    
    e_noise = estimated_signal - s_target
    
    si_sdr_value = 10 * torch.log10(torch.sum(s_target ** 2, dim=1) / torch.sum(e_noise ** 2, dim=1))
    
    return si_sdr_value


def lsd_batch(x_batch, y_batch, fs=16000, frame_size=0.02, frame_shift=0.02, start=0, cutoff_freq=0):
    frame_length = int(frame_size * fs)
    frame_step = int(frame_shift * fs)
   
    if isinstance(x_batch, np.ndarray):
        x_batch = torch.from_numpy(x_batch)
        y_batch = torch.from_numpy(y_batch)
   
    if x_batch.dim()==1:
        batch_size = 1
    ## 1 x 32000
    elif x_batch.dim()==2:
        x_batch=x_batch.unsqueeze(1)
    batch_size, _, signal_length = x_batch.shape
   
    if y_batch.dim()==1:
        y_batch=y_batch.reshape(batch_size,1,-1)
    elif y_batch.dim()==2:
        y_batch=y_batch.unsqueeze(1)
   
    lsd_values = []

    for i in range(batch_size):
        x = x_batch[i, 0, :].numpy()
        y = y_batch[i, 0, :].numpy()
 
        # STFT
        ## nfft//2 +1: freq len
        f_x, t_x, Zxx_x = stft(x, fs, nperseg=frame_length, noverlap=frame_length - frame_step, nfft=512)
        f_y, t_y, Zxx_y = stft(y, fs, nperseg=frame_length, noverlap=frame_length - frame_step, nfft=512)
       
        # Power spec
        power_spec_x = np.abs(Zxx_x) ** 2
        power_spec_y = np.abs(Zxx_y) ** 2
       
        # Log Power Spec
        log_spec_x = np.log10(power_spec_x + 1e-10)  # eps
        log_spec_y = np.log10(power_spec_y + 1e-10)

        if start or cutoff_freq:
            freq_len = log_spec_x.shape[0]
            max_freq = fs // 2
            start = int(start / max_freq * freq_len)
            freq_idx = int(cutoff_freq / max_freq * freq_len)
            log_spec_x = log_spec_x[start:freq_idx,:]
            log_spec_y = log_spec_y[start:freq_idx,:]

        #Spectral Mean
        lsd = np.sqrt(np.mean((log_spec_x - log_spec_y) ** 2, axis=0))
       
        #Frame mean
        mean_lsd = np.mean(lsd)
        lsd_values.append(mean_lsd)
   
    # Batch mean
    batch_mean_lsd = np.mean(lsd_values)
    # return log_spec_x, log_spec_y
    return batch_mean_lsd

## 언젠가는 분석해볼 것
def lsd(self, est, target):
        lsd = torch.log10(target**2 / ((est + 1e-12) ** 2) + 1e-12) ** 2
        lsd = torch.mean(torch.mean(lsd, dim=3) ** 0.5, dim=2)
        return lsd[..., None, None]

def draw_spec(x,
              figsize=(10, 6), title='', n_fft=2048,
              win_len=1024, hop_len=256, sr=16000, cmap='inferno',
              vmin=-50, vmax=40, use_colorbar=True,
              ylim=None,
              title_fontsize=10,
              label_fontsize=8,
                return_fig=False,
                save_fig=False, save_path=None):
    fig = plt.figure(figsize=figsize)
    stft = librosa.stft(x, n_fft=n_fft, hop_length=hop_len, win_length=win_len)
    stft = 20 * np.log10(np.clip(np.abs(stft), a_min=1e-8, a_max=None))

    plt.imshow(stft,
               aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
               origin='lower', extent=[0, len(x) / sr, 0, sr//2])

    if use_colorbar:
        plt.colorbar()

    plt.xlabel('Time (s)', fontsize=label_fontsize)
    plt.ylabel('Frequency (Hz)', fontsize=label_fontsize)

    if ylim is None:
        ylim = (0, sr / 2)
    plt.ylim(*ylim)

    plt.title(title, fontsize=title_fontsize)
    
    if save_fig and save_path:
        plt.savefig(f"{save_path}.png")
    
    if return_fig:
        plt.close()
        return fig
    else:
        # plt.close()
        plt.show()
        return stft
        
# import random
# from train import RTBWETrain
# from torch.utils.data import Subset
# from NewSEANet import NewSEANet
# from SEANet import SEANet

# def SmallDataset(dataloader, num_samples):
#     dataset = dataloader.dataset
#     total_samples = len(dataset)
#     indices = list(range(total_samples))
#     random.shuffle(indices)
#     random_indices = indices[:num_samples]
    
#     subset = Subset(dataset, random_indices)
#     return subset

# def load_generator(config, ckpt_path, device='cuda'):
#     if ckpt_path.endswith('.ckpt'):
#         # Load the full model from RTBWETrain (old version)
#         model = RTBWETrain.load_from_checkpoint(checkpoint_path=ckpt_path, config=config).to(device).eval()
#         generator = model.generator
#         epoch, pesq_score, lsd = model.current_epoch, None, None  # Placeholder values, modify as needed
#         print("***************")
#         print(f"Checkpoint loaded from: {ckpt_path}")
#         print(f"Generator: {type(generator).__name__}")

#     elif ckpt_path.endswith('.pth'):
#         # Load the generator from the new version checkpoint
#         generator_type = config['model']['generator']
#         if generator_type == 'SEANet':
#             generator = SEANet(min_dim=8, causality=True)
#             print("***************")
#             print("SEANet Generator")
#         elif generator_type == 'NewSEANet':
#             generator = NewSEANet(min_dim=8, 
#                                   kmeans_model_path=config['model']['kmeans_path'],
#                                   modelname=config['model']['sslname'],
#                                   causality=True)
#             print("***************")
#             print(f"NewSEANet Generator")
#             print(f"SSL Model: {config['model']['sslname']}")
#             print(f"KMeans Model: {os.path.splitext(os.path.basename(config['model']['kmeans_path']))[0]}")
#         else:
#             raise ValueError(f"Unsupported generator type: {generator_type}")
        
#         # Load the checkpoint
#         checkpoint = torch.load(ckpt_path, map_location=torch.device(device))
#         # Load the generator state dict
#         generator.load_state_dict(checkpoint['generator_state_dict'])
#         # Get additional information if needed
#         epoch = checkpoint['epoch']
#         pesq_score = checkpoint['pesq_score']
#         lsd = checkpoint.get('lsd', 0)
#         print(f"Checkpoint loaded from: {ckpt_path}")
#         print(f"Epoch: {epoch}, PESQ Score: {pesq_score:.2f}, LSD: {lsd:.2f}")
#         print("***************")
#     else:
#         raise ValueError(f"Unsupported checkpoint file extension: {ckpt_path}")

#     return generator

from pystoi import stoi
def py_stoi(clean_audio, processed_audio, sample_rate=16000):

    # Calculate STOI
    stoi_score = stoi(clean_audio, processed_audio, sample_rate, extended=False)
    return stoi_score


# adapted from
# https://github.com/kan-bayashi/ParallelWaveGAN/tree/master/parallel_wavegan
"""PQMF module.

This module is based on `Near-perfect-reconstruction pseudo-QMF banks`_.

.. _`Near-perfect-reconstruction pseudo-QMF banks`:
    https://ieeexplore.ieee.org/document/258122

"""
class PQMF(torch.nn.Module):
    def __init__(self, N=4, taps=62, cutoff=0.15, beta=9.0):
        super(PQMF, self).__init__()

        self.N = N
        self.taps = taps
        self.cutoff = cutoff
        self.beta = beta
        
        # scipy FIR filter coefficients
        QMF = sig.firwin(taps + 1, cutoff, window=('kaiser', beta))
        H = np.zeros((N, len(QMF)))
        G = np.zeros((N, len(QMF)))
        for k in range(N):
            constant_factor = (2 * k + 1) * (np.pi /
                                             (2 * N)) * (np.arange(taps + 1) -
                                                         ((taps - 1) / 2))  # TODO: (taps - 1) -> taps
            phase = (-1)**k * np.pi / 4
            # Analysis Filter
            H[k] = 2 * QMF * np.cos(constant_factor + phase)
            # Synthesis Filter
            G[k] = 2 * QMF * np.cos(constant_factor - phase)

        H = torch.from_numpy(H[:, None, :]).float()
        G = torch.from_numpy(G[None, :, :]).float()

        self.register_buffer("H", H)
        self.register_buffer("G", G)

        updown_filter = torch.zeros((N, N, N)).float()
        for k in range(N):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)
        self.N = N

        self.pad_fn = torch.nn.ConstantPad1d(taps // 2, 0.0)

    def forward(self, x):
        return self.analysis(x)

    def analysis(self, x):
        return F.conv1d(x, self.H, padding=self.taps // 2, stride=self.N)

    def synthesis(self, x):
        x = F.conv_transpose1d(x,
                               self.updown_filter * self.N,
                               stride=self.N)
        x = F.conv1d(x, self.G, padding=self.taps // 2)
        return x



def main():
    parser = argparse.ArgumentParser(description="Path for train test split")
    parser.add_argument("--path", type=str, help="Path for dataset")
    args = parser.parse_args()
    path_into_traintest(args.path)

if __name__ == "__main__":
    main()

