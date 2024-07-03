import torchaudio as ta
import pytorch_lightning as pl
import torchaudio.transforms as T
import numpy as np
import random
import torch
import sys
import torch.nn.functional as F
# from train import RTBWETrain
# from datamodule import *
from utils import *
from torch.utils.data import Dataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure reproducibility on cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("SEED!:",seed)

class CustomDataset(Dataset):
    def __init__(self, path_dir_nb, path_dir_wb, seg_len, iscodec=True, mode="train"):
        self.path_dir_nb = path_dir_nb
        self.path_dir_wb = path_dir_wb
        self.seg_len = seg_len
        self.mode = mode
        self.iscodec = iscodec

        set_seed(seed=42)

        paths_wav_wb = get_audio_paths(self.path_dir_wb, file_extensions='.flac')
        if self.iscodec:
            paths_wav_nb = get_audio_paths(self.path_dir_nb, file_extensions='.wav')
        else:
            paths_wav_nb = get_audio_paths(self.path_dir_nb, file_extensions='.flac')

        print(f"LR {len(paths_wav_nb)} and HR {len(paths_wav_wb)} file numbers loaded!")

        if len(paths_wav_wb) != len(paths_wav_nb):
            sys.exit(f"Error: LR {len(paths_wav_nb)} and HR {len(paths_wav_wb)} file numbers are different!")

        self.filenames = [(path_wav_wb, path_wav_nb) for path_wav_wb, path_wav_nb in zip(paths_wav_wb, paths_wav_nb)]
        print(f"{mode}: {len(self.filenames)} files loaded")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path_wav_wb, path_wav_nb = self.filenames[idx]

        wav_nb, sr_nb = ta.load(path_wav_nb)
        wav_wb, sr_wb = ta.load(path_wav_wb)

        wav_wb = wav_wb.view(1, -1)
        wav_nb = wav_nb.view(1, -1)

        if self.seg_len > 0 and self.mode == "train":
            duration = int(self.seg_len * 16000)
            sig_len = wav_wb.shape[-1]

            seed = idx
            np.random.seed(seed)

            t_start = np.random.randint(low=0, high=np.max([1, sig_len - duration - 2]), size=1)[0]
            if t_start % 2 == 1:
                t_start -= 1
            t_end = t_start + duration

            if self.iscodec:
                wav_nb = wav_nb.repeat(1, t_end // sig_len + 1)[..., t_start:t_end]
            else:
                wav_nb = wav_nb.repeat(1, t_end // sig_len + 1)[..., t_start // 2:t_end // 2]
            wav_wb = wav_wb.repeat(1, t_end // sig_len + 1)[..., t_start:t_end]

            wav_nb = self.ensure_length(wav_nb, sr_nb * self.seg_len)
            wav_wb = self.ensure_length(wav_wb, sr_wb * self.seg_len)

        elif self.mode == "val":
            min_len = min(wav_wb.shape[-1], wav_nb.shape[-1])
            wav_nb = self.ensure_length(wav_nb, min_len)
            wav_wb = self.ensure_length(wav_wb, min_len)

        else:
            sys.exit(f"unsupported mode! (train/val)")

        return wav_nb, wav_wb, get_filename(path_wav_wb)[0]

    @staticmethod
    def ensure_length(wav, target_length):
        if wav.shape[1] < target_length:
            pad_size = target_length - wav.shape[1]
            wav = F.pad(wav, (0, pad_size))
        elif wav.shape[1] > target_length:
            wav = wav[:, :target_length]
        return wav
    
class PQMFDataset(Dataset):
    # base_dir : VCTK-PQMF/16k
    def __init__(self, base_dir, seg_len, mode="train"):
        self.path_dir_lf = os.path.join(base_dir, mode, 'lf')
        self.path_dir_hf = os.path.join(base_dir, mode, 'hf')
        self.seg_len = seg_len
        self.mode = mode

        paths_wav_hf = get_audio_paths(self.path_dir_hf, file_extensions='.flac')
        paths_wav_lf = get_audio_paths(self.path_dir_lf, file_extensions='.flac')

        print(f"LR {len(paths_wav_lf)} and HR {len(paths_wav_hf)} file numbers loaded!")

        if len(paths_wav_hf) != len(paths_wav_lf):
            sys.exit(f"Error: LR {len(paths_wav_lf)} and HR {len(paths_wav_hf)} file numbers are different!")

        self.filenames = [(path_wav_hf, path_wav_lf) for path_wav_hf, path_wav_lf in zip(paths_wav_hf, paths_wav_lf)]
        print(f"{mode}: {len(self.filenames)} files loaded")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path_wav_hf, path_wav_lf = self.filenames[idx]

        wav_nb, sr_nb = ta.load(path_wav_lf)
        wav_wb, sr_wb = ta.load(path_wav_hf)

        wav_wb = wav_wb.view(1, -1)
        wav_nb = wav_nb.view(1, -1)

        if self.seg_len > 0 and self.mode == "train":
            # sr: 8000 for PQMF
            duration = int(self.seg_len * 8000)
            sig_len = wav_wb.shape[-1]

            t_start = np.random.randint(low=0, high=np.max([1, sig_len - duration - 2]), size=1)[0]
            if t_start % 2 == 1:
                t_start -= 1
            t_end = t_start + duration

            # Repeat
            wav_nb = wav_nb.repeat(1, t_end // sig_len + 1)[..., t_start:t_end]
            wav_wb = wav_wb.repeat(1, t_end // sig_len + 1)[..., t_start:t_end]

        elif self.mode == "test":
            min_len = min(wav_wb.shape[-1], wav_nb.shape[-1])
            wav_nb = self.ensure_length(wav_nb, min_len)
            wav_wb = self.ensure_length(wav_wb, min_len)
            pass

        else:
            sys.exit(f"unsupported mode! (train/test)")

        return wav_nb, wav_wb, get_filename(path_wav_hf)[0]

    @staticmethod
    def ensure_length(wav, target_length):
        if wav.shape[1] < target_length:
            pad_size = target_length - wav.shape[1]
            wav = F.pad(wav, (0, pad_size))
        elif wav.shape[1] > target_length:
            wav = wav[:, :target_length]
        return wav
    
def collate_fn_gt(batch, pqmf):
    """
    Collate function to make a ground-truth synthesized speech
    """
    wav_nbs, wav_wbs, filenames = zip(*batch)

    ## Concatenate LF and HF bands
    wav_nbs = torch.cat(wav_nbs, dim=0)
    wav_wbs = torch.cat(wav_wbs, dim=0)
    
    subbands = torch.stack((wav_nbs, wav_wbs), dim=1)
    wav_gt = pqmf.synthesis(subbands).squeeze(1)
    
    return wav_nbs, wav_wbs, wav_gt, list(filenames)