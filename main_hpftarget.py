import torchaudio as ta
import pytorch_lightning as pl
import torchaudio.transforms as T
import torch
import numpy as np
import yaml
import mir_eval
import gc

import warnings
# from train import RTBWETrain
# from datamodule import *
from utils import *

from tqdm import tqdm
import wandb
from pesq import pesq
from pystoi import stoi
import random
from torch.utils.data import Subset
import soundfile as sf
from datetime import datetime
import sys
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from SEANet import SEANet
from NewSEANet import NewSEANet
from MelGAN import Discriminator_MelGAN
from MBSTFTD import MultiBandSTFTDiscriminator

from dataset import HighTargetDataset
# from ssdiscriminatorblock import MultiBandSTFTDiscriminator


DEVICE = f'cuda' if torch.cuda.is_available() else 'cpu'

## Dictionary to store all models and information
TPARAMS = {}
NOTES = 'wavlm16_hightarget'
config = yaml.load(open("./config_wavlm16.yaml", 'r'), Loader=yaml.FullLoader)
START_DATE = NOTES +'_' + datetime.now().strftime("%Y%m%d-%H%M%S")

def wandb_log(loglist, epoch, note):
    for key, val in loglist.items():
        if isinstance(val, torch.Tensor):
            item = val.cpu().detach().numpy()
        else:
            item = val
        try:
            if isinstance(item, float):
                log = item
            elif isinstance(item, plt.Figure):
                log = wandb.Image(item)
                plt.close(item)
            elif item.ndim in [2, 3]:  # 이미지 데이터
                log = wandb.Image(item, caption=f"{note.capitalize()} {key.capitalize()} Epoch {epoch}")
            elif item.ndim == 1:  # 오디오 데이터d
                log = wandb.Audio(item, sample_rate=16000, caption=f"{note.capitalize()} {key.capitalize()} Epoch {epoch}")
            else:
                log = item
        except Exception as e:
            print(f"Failed to log {key}: {e}")
            log = item

        wandb.log({
            f"{note.capitalize()} {key.capitalize()}": log,
        }, step=epoch)

#########################
def train_step(train_parameters):
    train_parameters['generator'].train()
    train_parameters['discriminator'].train()
    result = {}
    result['loss_G'] = 0
    result['loss_D'] = 0
    result['FM_loss'] = 0  
    result['Mel_loss'] = 0 

    train_bar = tqdm(train_parameters['train_dataloader'], desc="Train", position=1, leave=False, disable=False)
    i = 0

    # Train DataLoader Loop
    # Epoch 1개만큼 train
    for gt, target, lr, _ in train_bar:
        i += 1
        lr = lr.to(DEVICE)
        gt = gt.to(DEVICE)
        target = target.to(DEVICE)
        high_extend = train_parameters['generator'](lr, gt)
        
        '''
        Train Generator
        '''        
        train_parameters['optim_G'].zero_grad()
        _, loss_GAN, loss_FM, loss_mel = train_parameters['discriminator'].loss_G(high_extend, target)
        ## MS Mel Loss
        loss_G = loss_GAN + 100*loss_FM + loss_mel
        loss_G.mean().backward()
        train_parameters['optim_G'].step()

        '''
        Discriminator
        '''
        train_parameters['optim_D'].zero_grad()
        loss_D = train_parameters['discriminator'].loss_D(high_extend, target)
        loss_D.mean().backward()
        train_parameters['optim_D'].step()
        
        result['loss_G'] += loss_G.item()
        result['loss_D'] += loss_D.item()
        result['FM_loss'] += loss_FM.item()  # FM loss 추가
        result['Mel_loss'] += loss_mel.item()  # Mel loss 추가

        train_bar.set_postfix({
                'Loss G': f'{loss_G.item():.2f}',
                'FM loss': f'{loss_FM.item():.2f}',
                'Mel loss': f'{loss_mel.item():.2f}',
                'Loss D': f'{loss_D.item():.2f}'
            })
        
        del lr, gt, target, loss_GAN, loss_FM, loss_mel, loss_D
        gc.collect()

    train_bar.close()
    result['loss_G'] /= len(train_parameters['train_dataloader'])
    result['loss_D'] /= len(train_parameters['train_dataloader'])
    result['FM_loss'] /= len(train_parameters['train_dataloader'])  
    result['Mel_loss'] /= len(train_parameters['train_dataloader']) 


    return result

def test_step(test_parameters, store_lr_hr=False):
    test_parameters['generator'].eval()
    test_parameters['discriminator'].eval()
    result = {}
    result['loss_G'] = 0
    result['loss_D'] = 0
    result['FM_loss'] = 0  
    result['Mel_loss'] = 0  
    # result['PESQ'] = 0  
    result['LSD'] = 0
    result['LSD_H'] = 0
    result['STOI'] = 0

    test_bar = tqdm(test_parameters['val_dataloader'], desc='Validation', position=1, leave=False, disable=False)
    i = 0
    # total_pesq = 0
    total_lsd = 0
    total_lsd_h = 0
    total_stoi = 0

    # Test DataLoader Loop
    with torch.no_grad():
        for gt, target, lr, _ in test_bar:
            i += 1
            lr = lr.to(DEVICE)
            gt = gt.to(DEVICE)
            target = target.to(DEVICE)
            high_extend = test_parameters['generator'](lr, gt)
            
            _, loss_GAN, loss_FM, loss_mel = test_parameters['discriminator'].loss_G(high_extend, target)
            loss_G = loss_GAN + 100*loss_FM + loss_mel
            loss_D = test_parameters['discriminator'].loss_D(high_extend, target)
            
            result['loss_G'] += loss_G.item()
            result['loss_D'] += loss_D.item()
            result['FM_loss'] += loss_FM.item()
            result['Mel_loss'] += loss_mel.item()

            # pesq_score =  pesq(fs=16000, ref=hr.squeeze().cpu().numpy(), deg=bwe.squeeze().cpu().numpy(), mode="wb")
            # total_pesq += pesq_score
            
            batch_lsd = lsd_batch(x_batch=gt.cpu(), y_batch=high_extend.cpu())
            total_lsd += batch_lsd

            batch_lsd_h = lsd_batch(x_batch=gt.cpu(), y_batch=high_extend.cpu(), start=3000, cutoff_freq=8000)
            total_lsd_h += batch_lsd_h

            stoi_score = py_stoi(gt.squeeze().cpu().numpy(), high_extend.squeeze().cpu().numpy(), sample_rate=16000)
            total_stoi += stoi_score

            test_bar.set_postfix({
                'Loss G': f'{loss_G.item():.2f}',
                'FM loss': f'{loss_FM.item():.2f}',
                'Mel loss': f'{loss_mel.item():.2f}',
                'Loss D': f'{loss_D.item():.2f}',
                # 'PESQ': f'{pesq_score:.2f}',
                'LSD_H': f'{batch_lsd_h:.2f}',
                'STOI': f'{stoi_score:.2f}'
            })

            if i == 2000 and store_lr_hr:  # For very first epoch
                result['audio_lr'] = lr.squeeze().cpu().numpy()
                result['audio_gt'] = gt.squeeze().cpu().numpy()
                result['audio_target'] = target.squeeze().cpu().numpy()
                result['spec_lr'] = draw_spec(lr.squeeze().cpu().numpy(),win_len=320, use_colorbar=False, hop_len=160, return_fig=True)
                result['spec_gt'] = draw_spec(gt.squeeze().cpu().numpy(),win_len=320, use_colorbar=False, hop_len=160, return_fig=True)
                result['spec_target'] = draw_spec(target.squeeze().cpu().numpy(),win_len=320, use_colorbar=False, hop_len=160, return_fig=True)
            if i == 2000:
                result['audio_bwe'] = high_extend.squeeze().cpu().numpy()
                result['spec_bwe'] = draw_spec(high_extend.squeeze().cpu().numpy(),win_len=320, hop_len=160, 
                                               use_colorbar=False, return_fig=True)

            del lr, gt, target, high_extend, loss_GAN, loss_FM, loss_mel, loss_D
            gc.collect()
            torch.cuda.empty_cache()


        test_bar.close()
        result['loss_G'] /= len(test_parameters['val_dataloader'])
        result['loss_D'] /= len(test_parameters['val_dataloader'])
        result['FM_loss'] /= len(test_parameters['val_dataloader'])  # FM loss 평균 계산
        result['Mel_loss'] /= len(test_parameters['val_dataloader'])  # Mel loss 평균 계산
        # result['PESQ'] = total_pesq / len(test_parameters['val_dataloader'])
        result['LSD'] = total_lsd / len(test_parameters['val_dataloader'])
        result['LSD_H'] = total_lsd_h / len(test_parameters['val_dataloader'])
        result['STOI'] = total_stoi / len(test_parameters['val_dataloader'])
    return result

def main():
    ################ Read Config Files

    wandb.init(project='SSLBWE_phase2',
           entity='woongzip1',
           config=config,
           name=START_DATE,
           # mode='disabled',
           notes=NOTES)

    train_path_gt = "/ssd2/woongzip/Datasets/VCTK_0.92_crop/16k/train"
    train_path_target = "/ssd2/woongzip/Datasets/VCTK_0.92_crop/16k_recon/train"
    train_path_lr = "/ssd2/woongzip/Datasets/VCTK_0.92_crop/MP3_sr16_8kbps/train"

    val_path_gt = "/ssd2/woongzip/Datasets/VCTK_0.92_crop/16k/test"
    val_path_target = "/ssd2/woongzip/Datasets/VCTK_0.92_crop/16k_recon/test"
    val_path_lr = "/ssd2/woongzip/Datasets/VCTK_0.92_crop/MP3_sr16_8kbps/test"

    ################ Load Dataset
    train_dataset = HighTargetDataset(train_path_gt, train_path_target, train_path_lr, 
                                mode='train')
    valid_dataset = HighTargetDataset(val_path_gt, val_path_target, val_path_lr, 
                                mode='val')
    print(f'Train Dataset size: {len(train_dataset)} | Validation Dataset size: {len(valid_dataset)}\n')

    ################ Load DataLoader
    TPARAMS['train_dataloader'] = DataLoader(train_dataset, batch_size = 16, shuffle=True,
                                            num_workers=config['dataset']['num_workers'], prefetch_factor=2, persistent_workers=True)
    TPARAMS['val_dataloader'] = DataLoader(valid_dataset, batch_size = 1, shuffle=False,
                                            num_workers=config['dataset']['num_workers'], prefetch_factor=2, persistent_workers=True)
    print(f"DataLoader Loaded!: {len(TPARAMS['train_dataloader'])} | {len(TPARAMS['val_dataloader'])}")

    # TPARAMS['train_dataloader'] = SmallDataset(TPARAMS['train_dataloader'], 200)
    # TPARAMS['val_dataloader'] = SmallDataset(TPARAMS['val_dataloader'], 200)

    ################ Load Models
    warnings.filterwarnings("ignore", category=UserWarning, message="torch.nn.utils.weight_norm is deprecated")
    warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated")
    warnings.filterwarnings("ignore", message=".*cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR.*")

    gen_type = config['model']['generator']

    print("########################################")
    if gen_type == "SEANet":
        TPARAMS['generator'] = SEANet(min_dim=8, causality=True)
        print(f"SEANet Generator: \n"
    f"                  Disc: {config['model']['discriminator']}")
    elif gen_type == "NewSEANet":
        TPARAMS['generator'] = NewSEANet(min_dim=8, 
                                       kmeans_model_path=config['model']['kmeans_path'],
                                       modelname=config['model']['sslname'],
                                       causality=True)
        print(f"NewSEANeT + {config['model']['sslname']} Generator: \n"
    f"                  kmeans:{os.path.splitext(os.path.basename(config['model']['kmeans_path']))[0]} \n"
    f"                  Disc: {config['model']['discriminator']},\n"
    f"                  SSL model: {config['model']['sslname']}")
    else: 
        raise ValueError(f"Unsupported generator type: {gen_type}")

    disc_type = config['model']['discriminator']
    if disc_type == "MSD":
        TPARAMS['discriminator'] = Discriminator_MelGAN()
    elif disc_type == "MBSTFTD":
        discriminator_config = config['model']['MultiBandSTFTDiscriminator_config']
        TPARAMS['discriminator'] = MultiBandSTFTDiscriminator(
            C=discriminator_config['C'],
            n_fft_list=discriminator_config['n_fft_list'],
            hop_len_list=discriminator_config['hop_len_list'],
            bands=discriminator_config['band_split_ratio'],
            ms_mel_loss_config=config['model']['ms_mel_loss_config']
        )
    else:
        raise ValueError(f"Unsupported discriminator type: {disc_type}")
    print("########################################")
    ################ Load Optimizers
    TPARAMS['optim_G'] = torch.optim.Adam(TPARAMS['generator'].parameters(), lr=config['optim']['learning_rate'], 
                                          betas=(config['optim']['B1'],config['optim']['B2']))
    TPARAMS['optim_D'] = torch.optim.Adam(TPARAMS['discriminator'].parameters(), config['optim']['learning_rate'], 
                                          betas=(config['optim']['B1'],config['optim']['B2']))
    
    ################ Load Checkpoint if available
    start_epoch = 1
    best_pesq = -1

    if config['train']['ckpt']:
        checkpoint_path = config['train']['ckpt_path']
        if os.path.isfile(checkpoint_path):
            start_epoch, best_pesq = load_checkpoint(TPARAMS['generator'], TPARAMS['discriminator'],
                                                     TPARAMS['optim_G'], TPARAMS['optim_D'], checkpoint_path)
        else:
            print(f"Checkpoint file not found at {checkpoint_path}. Starting training from scratch.")

    ################ Training Code
    print('Train Start!!')
    BAR = tqdm(range(start_epoch, config['train']['max_epochs'] + 1), position=0, leave=True)
    TPARAMS['generator'].to(DEVICE)
    TPARAMS['discriminator'].to(DEVICE)
    best_LSD = 1e10

    store_lr_hr = True # flag
    for epoch in BAR:
        # set_seed(epoch+42)

        TPARAMS['current_epoch'] = epoch
        train_result = train_step(TPARAMS)
        wandb_log(train_result, epoch, 'train')

        if epoch % config['train']['val_epoch'] == 0:
            # Validation step
            val_result = test_step(TPARAMS, store_lr_hr)
            wandb_log(val_result, epoch, 'val')

            if store_lr_hr:
                store_lr_hr = False
            # Best LSD Model
            if val_result['LSD_H'] < best_LSD:
                best_lsd = val_result['LSD_H']
                save_checkpoint(TPARAMS['generator'], TPARAMS['discriminator'], epoch, best_lsd, config)

            desc = (f"Epoch [{epoch}/{config['train']['max_epochs']}] "
                    f"Loss G: {train_result['loss_G']:.2f}, "
                    f"FM Loss: {train_result['FM_loss']:.2f}, "
                    f"Mel Loss: {train_result['Mel_loss']:.2f}, "
                    f"Loss D: {train_result['loss_D']:.2f}, "
                    f"Val Loss G: {val_result['loss_G']:.2f}, "
                    f"Val FM Loss: {val_result['FM_loss']:.2f}, "
                    f"Val Mel Loss: {val_result['Mel_loss']:.2f}, "
                    f"Val Loss D: {val_result['loss_D']:.2f}, "
                    f"LSD: {val_result['LSD']:.2f}"
                    )
            
        else:
            desc = (f"Epoch [{epoch}/{config['train']['max_epochs']}] "
                    f"Loss G: {train_result['loss_G']:.2f}, "
                    f"FM Loss: {train_result['FM_loss']:.2f}, "
                    f"Mel Loss: {train_result['Mel_loss']:.2f}, "
                    f"Loss D: {train_result['loss_D']:.2f}"
                    )
        BAR.set_description(desc)

    gc.collect()
    final_epoch = config['train']['max_epochs']
    save_checkpoint(TPARAMS['generator'], TPARAMS['discriminator'], final_epoch, val_result['LSD_H'], config)

def save_checkpoint(generator, discriminator, epoch, lsd_h, config):
    checkpoint_dir = config['train']['ckpt_save_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}_lsdH_{lsd_h:.2f}.pth")
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': TPARAMS['optim_G'].state_dict(),  # Save optimizer state
        'optimizer_D_state_dict': TPARAMS['optim_D'].state_dict(),  # Save optimizer state
        'lsd_h': lsd_h,
    }, checkpoint_path)
    print(f"Checkpoint saved at: {checkpoint_path}")

def load_checkpoint(generator, discriminator, optimizer_G, optimizer_D, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_pesq = checkpoint['pesq_score']
    
    if 'optimizer_G_state_dict' in checkpoint and 'optimizer_D_state_dict' in checkpoint:
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

    return start_epoch, best_pesq

def SmallDataset(dataloader, num_samples):
    dataset = dataloader.dataset
    total_samples = len(dataset)
    indices = list(range(total_samples))
    random.shuffle(indices)
    random_indices = indices[:num_samples]
    
    subset = Subset(dataset, random_indices)
    return subset

if __name__ == "__main__":
    main()