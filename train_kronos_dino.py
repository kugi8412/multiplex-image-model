import os
import sys

import comet_ml  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from copy import deepcopy

# Imports from your ecosystem
from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler
from multiplex_model.utils import init_experiment, finish_experiment, get_run_name

# Imports from KRONOS (requires 'kronos' folder next to the script)
from multiplex_model.kronos.vision_transformer import vit_small, vit_large
from multiplex_model.kronos.dino_head import DINOHead


# ==========================================
# 1. KRONOS DINO ARCHITECTURE
# ==========================================
class KronosDINO(nn.Module):
    """Wrapper combining ViT Backbone with DINOHead projection head."""
    def __init__(self, backbone, embed_dim, out_dim):
        super().__init__()
        self.backbone = backbone
        self.head = DINOHead(
            in_dim=embed_dim,
            out_dim=out_dim,
            hidden_dim=2048,
            bottleneck_dim=256,
            nlayers=3,
        )

    def forward(self, x_list, marker_ids):
        """Student/Teacher forward pass for multiple crops."""
        masks_list = [None] * len(x_list)
        marker_ids_list = [marker_ids for _ in x_list]

        features = self.backbone.forward_features_list(x_list, masks_list, marker_ids_list)
        cls_tokens = torch.cat([f["x_norm_clstoken"] for f in features])
        
        return self.head(cls_tokens)


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
        
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2) # Teacher always gets 2 global crops

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
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / (len(teacher_output))
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0):
    warmup_iters = warmup_epochs * niter_per_ep
    warmup_schedule = np.array([])
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(0, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    return schedule


# ==========================================
# 2. MULTI-CROP AUGMENTATION & COLLATE FUNC
# ==========================================
class SpatialProteomicsMultiCrop:
    """Crops image into 2 global and N local crops. No pixel alterations (Color Jitter)."""
    def __init__(self, global_scale, local_scale, global_size, local_size, local_crops_num):
        self.global_transform = Compose([
            RandomResizedCrop(global_size, scale=global_scale, interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
        ])
        self.local_crops_num = local_crops_num
        self.local_transform = Compose([
            RandomResizedCrop(local_size, scale=local_scale, interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
        ])

    def __call__(self, img):
        crops = []
        crops.append(self.global_transform(img))
        crops.append(self.global_transform(img))
        for _ in range(self.local_crops_num):
            crops.append(self.local_transform(img))
        return crops

def dino_collate_fn(batch):
    """Groups crop lists into a proper Batch format."""
    num_crops = len(batch[0][0])
    collated_crops = []
    for i in range(num_crops):
        collated_crops.append(torch.stack([item[0][i] for item in batch]))
        
    channel_ids = torch.stack([item[1] for item in batch])
    panel_idxs = [item[2] for item in batch]
    img_paths = [item[3] for item in batch]
    
    return collated_crops, channel_ids, panel_idxs, img_paths


# ==========================================
# 3. MAIN TRAINING LOOP
# ==========================================
def main():
    config_path = sys.argv[1]
    yaml = YAML(typ="safe")
    with open(config_path, "r") as f:
        config = yaml.load(f)

    device = torch.device(config.get("device", "cuda"))
    print(f"Using device: {device}")

    # Data Configuration
    PANEL_CONFIG = YAML().load(open(config['panel_config_path']))
    TOKENIZER = YAML().load(open(config['tokenizer_config_path']))

    train_transform = SpatialProteomicsMultiCrop(
        global_scale=config['global_crops_scale'],
        local_scale=config['local_crops_scale'],
        global_size=config['global_crops_size'],
        local_size=config['local_crops_size'],
        local_crops_num=config['local_crops_number']
    )

    train_dataset = DatasetFromTIFF(
        panels_config=PANEL_CONFIG,
        split="train",
        marker_tokenizer=TOKENIZER,
        transform=train_transform,
        use_preprocessing=False, 
        use_median_denoising=False,
        use_butterworth_filter=True,
        use_minmax_normalization=False,
        use_clip_normalization=True,
        file_extension="npy",
    )

    train_batch_sampler = PanelBatchSampler(train_dataset, config['batch_size'])

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=config['num_workers'],
        collate_fn=dino_collate_fn, # <--- Required for Multi-Crop
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # KRONOS Initialization (Student and Teacher)
    num_markers = len(TOKENIZER)
    if config['model_name'] == 'vits16':
        backbone_student = vit_small(patch_size=16, num_markers=num_markers, drop_path_rate=config['drop_path_rate'])
        embed_dim = 384
    else:
        backbone_student = vit_large(patch_size=16, num_markers=num_markers, drop_path_rate=config['drop_path_rate'])
        embed_dim = 1024

    student = KronosDINO(backbone_student, embed_dim, config['out_dim']).to(device)
    teacher = deepcopy(student).to(device)
    
    # Freeze Teacher
    for param in teacher.parameters():
        param.requires_grad = False

    # Loss and Optimizer
    dino_loss = DINOLoss(
        out_dim=config['out_dim'],
        ncrops=config['global_crops_number'] + config['local_crops_number'],
        warmup_teacher_temp=config['warmup_teacher_temp'],
        teacher_temp=config['teacher_temp'],
        warmup_teacher_temp_epochs=config['warmup_teacher_temp_epochs'],
        nepochs=config['epochs']
    ).to(device)

    optimizer = optim.AdamW(student.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    niter_per_ep = len(train_dataloader)
    lr_schedule = cosine_scheduler(config['lr'], config['final_lr'], config['epochs'], niter_per_ep, config['warmup_epochs'])
    wd_schedule = cosine_scheduler(config['weight_decay'], config['weight_decay_final'], config['epochs'], niter_per_ep)
    momentum_schedule = cosine_scheduler(config['teacher_momentum'], 1.0, config['epochs'], niter_per_ep)

    # Comet Initialization
    init_experiment(config)
    run_name = get_run_name()

    checkpoints_path = config.get("checkpoints_dir", "checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)

    print("Starting DINOv2 KRONOS training...")
    scaler = torch.amp.GradScaler('cuda')
    
    # Enable anomaly detection (optional, good for debugging AMP)
    # torch.autograd.set_detect_anomaly(True)

    # ================= TRAINING LOOP =================
    for epoch in range(config['epochs']):
        student.train()
        running_loss = 0.0
        
        for batch_idx, (crops, channel_ids, panel_idx, img_path) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            global_step = niter_per_ep * epoch + batch_idx
            
            # Update schedulers
            for _, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[global_step]
                param_group["weight_decay"] = wd_schedule[global_step]

            # Fetch momentum for current step
            m = momentum_schedule[global_step]

            # Move to GPU
            crops = [crop.to(device, non_blocking=True) for crop in crops]
            channel_ids = channel_ids.to(device, non_blocking=True)

            # --- AUTOCAST CONTEXT ---
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                student_output = student(crops, channel_ids) 
                
                with torch.no_grad():
                    teacher_output = teacher(crops[:config['global_crops_number']], channel_ids) 

                # Loss computation
                loss = dino_loss(student_output, teacher_output, epoch)
            
            # Scale loss for gradient accumulation to prevent exploding gradients
            accumulated_loss = loss / config['gradient_accumulation_steps']
            scaler.scale(accumulated_loss).backward()
            
            # Perform optimizer step only after specific accumulation steps
            if ((batch_idx + 1) % config['gradient_accumulation_steps'] == 0) or ((batch_idx + 1) == len(train_dataloader)):
                
                # Unscale and clip gradients (from Table S24: max norm 3)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), config['clip_grad'])
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Update Teacher (EMA) ONLY when Student actually changes weights
                with torch.no_grad():
                    for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            # Record unscaled loss for logging
            running_loss += loss.item()

            # Comet.ml iteration logging
            if (batch_idx + 1) % 10 == 0:
                if comet_ml.get_global_experiment() is not None:
                    comet_ml.get_global_experiment().log_metrics({
                        "train/dino_loss": loss.item(),
                        "train/lr": lr_schedule[global_step],
                        "train/weight_decay": wd_schedule[global_step],
                        "train/teacher_momentum": m,
                    }, step=global_step)

        # End of epoch summary
        epoch_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch} - Avg Loss: {epoch_loss:.4f}")
        
        # Save Checkpoints (Saving the student - this is our final model)
        if (epoch + 1) % config.get("save_checkpoint_freq", 5) == 0:
            torch.save({
                "student_state_dict": student.state_dict(),
                "teacher_state_dict": teacher.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            }, f"{checkpoints_path}/kronos_dino-{run_name}-epoch_{epoch}.pth")

    print("Training finished!")
    torch.save(student.state_dict(), f"{checkpoints_path}/kronos_dino-{run_name}-final.pth")
    finish_experiment()

if __name__ == "__main__":
    main()
