#!/usr/bin/env python
# -*- coding: utf-8 -*-
# train_kronos_immuvis_v3.py


import os
import sys

import comet_ml  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomRotation,
)

# Imports from ImmuVis
from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler
from multiplex_model.utils import init_experiment, finish_experiment, get_run_name
from multiplex_model.modules.immuvis import Hyperkernel

# Imports from KRONOS
from multiplex_model.kronos.vision_transformer import Block, MemEffAttention
from multiplex_model.kronos.dino_head import DINOHead

# ------------------------------------------
# 1. KRONOS-IMMUVIS DINOv2 ARCHITECTURE
# ------------------------------------------
class ImmuvisDINO(nn.Module):
    """
    Prawdziwy model DINOv2 dla danych ImmuVis.
    Brak dekodera. Zwraca wielowymiarowy wektor cech biologicznych.
    """
    def __init__(self, num_markers, embed_dim=768, depth=12, num_heads=12, patch_size=16, out_dim=65536):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # 1. STEM: Hyperkernel ImmuVisa
        self.hyperkernel = Hyperkernel(
            num_channels=num_markers,
            input_dim=1,
            embedding_dim=embed_dim,
            module_type="encoder",
            kernel_size=patch_size, 
            stride=patch_size,
            padding=0
        )
        
        # 2. TRANSFORMER: KRONOS ViT
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024 + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True, attn_class=MemEffAttention)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # 3. DINO head: Linear classification for CLS
        self.head = DINOHead(in_dim=embed_dim, out_dim=out_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x, channel_ids):
        B, C, H, W = x.shape
        h_p, w_p = H // self.patch_size, W // self.patch_size
        N = h_p * w_p
        
        # forward Hyperkernel
        x_hk = x.reshape(B * C, 1, H, W)
        x_enc = self.hyperkernel(x_hk, channel_ids) 
        x_enc = x_enc.flatten(2).transpose(1, 2)    
        
        # Tokenization spatial + CLS
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_enc = torch.cat((cls_tokens, x_enc), dim=1)
        x_enc = x_enc + self.pos_embed[:, :N+1, :]
        
        for blk in self.blocks:
            x_enc = blk(x_enc)
        x_enc = self.norm(x_enc)
        
        cls_feature = x_enc[:, 0] # CLS token
        return self.head(cls_feature)

# ------------------------------------------
# 2. DINO LOSS & MULTI-CROP AUGMENTATION
# ------------------------------------------
class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs, nepochs, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        
        # Warmup for teacher temperature
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    # Update_center for teacher validation
    def forward(self, student_output, teacher_output, epoch, update_center=True):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # Centre for teacher output
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue

                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        
        if update_center:
            self.update_center(teacher_output)
            
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True) / teacher_output.shape[0]
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

class MultiCropTransform:
    def __init__(self, global_size, local_size, global_scale, local_scale, local_crops_number):
        
        # Global teacher transformation
        self.global_transform = Compose([
            RandomResizedCrop(global_size, scale=global_scale, interpolation=InterpolationMode.BILINEAR),
            RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlip(p=0.5), 
            RandomVerticalFlip(p=0.5)
        ])
        
        # Local student transformation
        self.local_transform = Compose([
            RandomResizedCrop(local_size, scale=local_scale, interpolation=InterpolationMode.BILINEAR),
            RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlip(p=0.5), 
            RandomVerticalFlip(p=0.5)
        ])
        self.local_crops_number = local_crops_number

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
            
        crops = []
        crops.append(self.global_transform(img))
        crops.append(self.global_transform(img))
        
        # Student N crops
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(img))
            
        return crops


class DINODatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        img, channel_ids, panel_idx, img_path = self.base_dataset[idx]
        crops = self.transform(img) # retunr N images
        return crops, channel_ids, panel_idx, img_path

def dino_collate_fn(batch):
    num_crops = len(batch[0][0])
    crops = [torch.stack([item[0][i] for item in batch]) for i in range(num_crops)]
    channel_ids = torch.stack([item[1] for item in batch])
    return crops, channel_ids

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0):
    warmup_iters = warmup_epochs * niter_per_ep
    warmup_schedule = np.array([]) if warmup_epochs == 0 else np.linspace(0, base_value, warmup_iters)
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    return np.concatenate((warmup_schedule, schedule))

# ------------------------------------------
# Train DINOv2
# ------------------------------------------
def main():
    config_path = sys.argv[1]
    yaml = YAML(typ="safe")
    with open(config_path, "r") as f:
        config = yaml.load(f)

    device = torch.device(config.get("device", "cuda"))
    print(f"Using device: {device}")

    PANEL_CONFIG = YAML().load(open(config['panel_config_path']))
    TOKENIZER = YAML().load(open(config['tokenizer_config_path']))
    num_markers = len(TOKENIZER)

    dino_transform = MultiCropTransform(
        global_size=config['global_crops_size'], local_size=config['local_crops_size'],
        global_scale=config['global_crops_scale'], local_scale=config['local_crops_scale'],
        local_crops_number=config['local_crops_number']
    )

    # Butterworth for training + DINO augmentation for student
    train_dataset_base = DatasetFromTIFF(
        panels_config=PANEL_CONFIG, split="train", marker_tokenizer=TOKENIZER,
        transform=None, use_preprocessing=False, use_butterworth_filter=True, 
        use_clip_normalization=True, file_extension="npy"
    )
    train_dataset = DINODatasetWrapper(train_dataset_base, dino_transform)
    train_sampler = PanelBatchSampler(train_dataset_base, config['batch_size'])
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=config['num_workers'], collate_fn=dino_collate_fn, pin_memory=True)

    # Validation loop
    val_dataset_base = DatasetFromTIFF(
        panels_config=PANEL_CONFIG, split="test", marker_tokenizer=TOKENIZER,
        transform=None, use_preprocessing=False, use_butterworth_filter=True, 
        use_clip_normalization=True, file_extension="npy"
    )
    val_dataset = DINODatasetWrapper(val_dataset_base, dino_transform)
    val_sampler = PanelBatchSampler(val_dataset_base, config['batch_size'], shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=config['num_workers'], collate_fn=dino_collate_fn, pin_memory=True)

    # Teacher + Student
    student = ImmuvisDINO(num_markers=num_markers, patch_size=config['patch_size'], out_dim=config['out_dim']).to(device)
    teacher = ImmuvisDINO(num_markers=num_markers, patch_size=config['patch_size'], out_dim=config['out_dim']).to(device)
    
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    # DINO Loss
    dino_loss = DINOLoss(
        out_dim=config['out_dim'], ncrops=2 + config['local_crops_number'],
        warmup_teacher_temp=config['warmup_teacher_temp'], teacher_temp=config['teacher_temp'],
        warmup_teacher_temp_epochs=config['warmup_teacher_temp_epochs'], nepochs=config['epochs']
    ).to(device)

    optimizer = optim.AdamW(student.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    niter_per_ep = len(train_dataloader)
    lr_schedule = cosine_scheduler(config['lr'], config['final_lr'], config['epochs'], niter_per_ep, config['warmup_epochs'])
    wd_schedule = cosine_scheduler(config['weight_decay'], config['weight_decay_final'], config['epochs'], niter_per_ep)
    momentum_schedule = cosine_scheduler(config['teacher_momentum'], 1.0, config['epochs'], niter_per_ep)

    # Loading from checkpoint
    start_epoch = 0
    checkpoint_path = config.get("from_checkpoint", None)
    
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        student.load_state_dict(checkpoint["student_state_dict"])
        teacher.load_state_dict(checkpoint["teacher_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        dino_loss.center = checkpoint["dino_loss_center"].to(device)
        start_epoch = checkpoint["epoch"] + 1
        print(f"[SUCCESS]: Resumed training from epoch {start_epoch}")
    elif checkpoint_path is not None:
        print(f"[WARNING]: Checkpoint path {checkpoint_path} not found.")

    init_experiment(config)
    experiment = comet_ml.get_global_experiment()
    run_name = get_run_name()

    checkpoints_path = config.get("checkpoints_dir", "checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)

    print("Starting True DINOv2 KRONOS-IMMUVIS Training...")
    scaler = torch.amp.GradScaler('cuda')
    grad_accum_steps = config.get('gradient_accumulation_steps', 1)

    for epoch in range(start_epoch, config['epochs']):
        student.train()
        train_loss = 0.0
        
        for batch_idx, (crops, channel_ids) in enumerate(tqdm(train_dataloader, desc=f"Train Epoch {epoch}")):
            global_step = niter_per_ep * epoch + batch_idx
            
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[global_step]
                param_group["weight_decay"] = wd_schedule[global_step]

            crops = [crop.to(device, dtype=torch.float32, non_blocking=True) for crop in crops]
            channel_ids = channel_ids.to(device, dtype=torch.long, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    teacher_out = teacher(torch.cat(crops[:2]), channel_ids.repeat(2, 1))
                
                student_global = student(torch.cat(crops[:2]), channel_ids.repeat(2, 1))
                student_local = student(torch.cat(crops[2:]), channel_ids.repeat(len(crops)-2, 1))
                student_out = torch.cat([student_global, student_local])
                loss = dino_loss(student_out, teacher_out, epoch)
            
            accumulated_loss = loss / grad_accum_steps
            scaler.scale(accumulated_loss).backward()
            
            if ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(train_dataloader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), config.get('clip_grad', 1.0))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # EMA for student -> teacher
                with torch.no_grad():
                    m = momentum_schedule[global_step]
                    for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            train_loss += loss.item()

            if (batch_idx + 1) % 10 == 0 and experiment is not None:
                experiment.log_metrics({
                    "train/dino_loss": loss.item(),
                    "train/lr": lr_schedule[global_step],
                    "train/teacher_momentum": momentum_schedule[global_step],
                }, step=global_step)

        # Validation phase
        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for crops, channel_ids in tqdm(val_dataloader, desc=f"Val Epoch {epoch}"):
                crops = [crop.to(device, dtype=torch.float32, non_blocking=True) for crop in crops]
                channel_ids = channel_ids.to(device, dtype=torch.long, non_blocking=True)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    teacher_out = teacher(torch.cat(crops[:2]), channel_ids.repeat(2, 1))
                    
                    student_global = student(torch.cat(crops[:2]), channel_ids.repeat(2, 1))
                    student_local = student(torch.cat(crops[2:]), channel_ids.repeat(len(crops)-2, 1))
                    student_out = torch.cat([student_global, student_local])
                    loss = dino_loss(student_out, teacher_out, epoch, update_center=False)
                    
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        
        print(f"Epoch {epoch} | Train DINO Loss: {avg_train_loss:.4f} | Val DINO Loss: {avg_val_loss:.4f}")
        
        if experiment is not None:
            experiment.log_metrics({
                "epoch/dino_loss": avg_train_loss,
                "epoch/val_dino_loss": avg_val_loss
            }, epoch=epoch)

        if (epoch + 1) % config.get("save_checkpoint_freq", 10) == 0:
            torch.save({
                "student_state_dict": student.state_dict(),
                "teacher_state_dict": teacher.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dino_loss_center": dino_loss.center,
                "epoch": epoch,
            }, f"{checkpoints_path}/kronos_dino-{run_name}-epoch_{epoch}.pth")

    print("Training finished!")
    torch.save(student.state_dict(), f"{checkpoints_path}/kronos_dino-{run_name}-final.pth")
    finish_experiment()


if __name__ == "__main__":
    main()
