import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import time
import math
from torch.utils.tensorboard import SummaryWriter
import gc
from image_data_loader import ImageDataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
import argparse

# Import TPU-specific modules conditionally
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    HAS_TPU = True
except ImportError:
    HAS_TPU = False

def get_available_devices():
    """Get list of available devices for argparse choices"""
    devices = ['auto', 'cpu']
    
    # Check CUDA availability
    if torch.cuda.is_available():
        devices.extend([f'cuda:{i}' for i in range(torch.cuda.device_count())])
    
    # Check MPS availability
    if torch.backends.mps.is_available():
        devices.append('mps')
    
    # Check TPU availability
    try:
        import torch_xla.core.xla_model as xm
        if xm.xrt_world_size() > 0:
            devices.append('tpu')
    except ImportError:
        pass
    
    return devices

def init_accelerator(device_str: str = None, mixed_precision: str = 'no'):
    """Initialize accelerator with optional device override and mixed precision"""
    if device_str and device_str != 'auto':
        # Manual device override
        if device_str == 'cpu':
            device_placement_policy = 'cpu'
            mixed_precision = 'no'  # Force no mixed precision for CPU
        elif device_str.startswith('cuda'):
            device_placement_policy = 'cuda'
            multi_gpu = torch.cuda.device_count() > 1
        elif device_str == 'mps':
            device_placement_policy = 'mps'
            mixed_precision = 'no'  # Force no mixed precision for MPS
            multi_gpu = False
        elif device_str == 'tpu':
            device_placement_policy = 'tpu'
            if mixed_precision == 'no':
                mixed_precision = 'bf16'  # TPU supports bfloat16 by default
            multi_gpu = False
        else:
            raise ValueError(f"Unsupported device: {device_str}")
        
        accelerator = Accelerator(
            device_placement_policy=device_placement_policy,
            mixed_precision=mixed_precision,
            dispatch_batches=device_str == 'tpu',
            split_batches=multi_gpu,
            gradient_accumulation_steps=model_config.gradient_accumulation_steps if 'model_config' in globals() else 8
        )
    else:
        # Let Accelerate automatically choose the best device
        multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1
        accelerator = Accelerator(
            mixed_precision=mixed_precision,
            split_batches=multi_gpu,
            gradient_accumulation_steps=model_config.gradient_accumulation_steps if 'model_config' in globals() else 8
        )
    
    return accelerator, accelerator.device

# Initialize globals
accelerator = None
device = None

def initialize_globals(args):
    """Initialize global accelerator and device"""
    global accelerator, device
    accelerator, device = init_accelerator(args.device, args.mixed_precision)
    
    # Print distributed training info
    if accelerator.num_processes > 1:
        print(f"\nDistributed Training Configuration:")
        print(f"Number of GPUs: {accelerator.num_processes}")
        print(f"Process rank: {accelerator.process_index}")
        print(f"Local process rank: {accelerator.local_process_index}")
        print(f"Device placement: {accelerator.device}")
        if accelerator.distributed_type == "MULTI_GPU":
            print("Using DistributedDataParallel (DDP)")
    
    print(f"\nUsing device: {device}")
    print(f"Mixed precision mode: {args.mixed_precision}")
    return accelerator, device

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Transformer model for image pixel prediction')
    
    # Device and distributed training settings
    parser.add_argument('--device', type=str, default='auto', 
                       choices=get_available_devices(),
                       help='Device to use (auto, cpu, cuda:N, mps, or tpu)')
    parser.add_argument('--multi_gpu', action='store_true',
                       help='Enable multi-GPU training using DistributedDataParallel')
    parser.add_argument('--mixed_precision', type=str, default='no',
                       choices=['no', 'fp16', 'bf16'],
                       help='Mixed precision mode (no, fp16, or bf16)')
    
    # Image size parameters
    parser.add_argument('--image_height', type=int, default=256,
                       help='Height of the images to generate and train on')
    parser.add_argument('--image_width', type=int, default=256,
                       help='Width of the images to generate and train on')
    
    # Model architecture
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='Dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='Number of warmup steps')
    parser.add_argument('--clip_grad_norm', type=float, default=0.5, help='Gradient clipping norm')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='Number of steps to accumulate gradients')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--train_seq_length', type=int, default=1024,
                       help='Maximum sequence length to use during training')
    
    # Generation parameters
    parser.add_argument('--generate_every', type=int, default=10,
                       help='Generate sample images every N epochs')
    parser.add_argument('--num_samples', type=int, default=2,
                       help='Number of sample images to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for sampling (higher = more random)')
    parser.add_argument('--inference_seq_length', type=int, default=512,
                       help='Maximum sequence length to keep in memory during generation')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--save_dir', type=str, default='pixel_transformer_checkpoints', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Adjust batch size based on number of GPUs if multi-GPU is enabled
    if args.multi_gpu and torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print(f"\nUsing {n_gpus} GPUs")
            print(f"Batch size per GPU: {args.batch_size}")
            print(f"Total batch size: {args.batch_size * n_gpus}")
            print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
            print(f"Effective total batch size: {args.batch_size * n_gpus * args.gradient_accumulation_steps}")
    
    return args

def clear_memory():
    """Clear memory cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif HAS_TPU and device.type == 'xla':
        # Synchronize TPU operations
        xm.mark_step()
    gc.collect()

@dataclass
class ModelConfig:
    # Transformer parameters
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    
    # Image parameters
    image_size: tuple = (32, 32)  # (height, width)
    sequence_length: int = 24  # 24 bits for RGB
    train_seq_length: int = 1024  # Maximum sequence length during training
    inference_seq_length: int = 512  # Maximum sequence length during generation
    
    # Training parameters
    learning_rate: float = 0.0001
    batch_size: int = 2
    warmup_steps: int = 4000
    clip_grad_norm: float = 0.5
    gradient_accumulation_steps: int = 16
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]

class PixelTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input projection from 24-bit to d_model
        self.input_projection = nn.Linear(config.sequence_length, config.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.d_model, config.image_size[0] * config.image_size[1])
        
        # Transformer encoder with KV caching
        encoder_layers = []
        for _ in range(config.num_layers):
            layer = nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(
                    embed_dim=config.d_model,
                    num_heads=config.nhead,
                    dropout=config.dropout,
                    batch_first=True
                ),
                'norm1': nn.LayerNorm(config.d_model),
                'ff': nn.Sequential(
                    nn.Linear(config.d_model, config.dim_feedforward),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.dim_feedforward, config.d_model)
                ),
                'norm2': nn.LayerNorm(config.d_model),
                'dropout': nn.Dropout(config.dropout)
            })
            encoder_layers.append(layer)
        self.encoder_layers = nn.ModuleList(encoder_layers)
        
        # Output projection back to 24-bit
        self.output_projection = nn.Linear(config.d_model, config.sequence_length)
        
        # Initialize parameters
        self.init_weights()
        
        # KV cache
        self.kv_cache = None
    
    def init_weights(self) -> None:
        """Initialize weights using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def get_model_size(self) -> int:
        """Get total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    def init_kv_cache(self, batch_size: int):
        """Initialize KV cache for faster generation"""
        self.kv_cache = []
        for _ in range(len(self.encoder_layers)):
            layer_cache = {
                'key': torch.zeros(batch_size, 0, self.config.d_model, device=device),
                'value': torch.zeros(batch_size, 0, self.config.d_model, device=device)
            }
            self.kv_cache.append(layer_cache)
    
    def clear_kv_cache(self):
        """Clear the KV cache"""
        self.kv_cache = None
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, use_cache: bool = False) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            src: Tensor, shape [batch_size, seq_len, 24]
            src_mask: Optional mask for masked prediction
            use_cache: Whether to use KV cache during generation
        Returns:
            output: Tensor, shape [batch_size, seq_len, 24]
        """
        # Initialize KV cache if needed
        if use_cache and self.kv_cache is None:
            self.init_kv_cache(src.size(0))
        
        # Project input to d_model dimensions
        x = self.input_projection(src)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Process through encoder layers with KV caching
        for i, layer in enumerate(self.encoder_layers):
            # Self attention with optional KV cache
            if use_cache:
                # Compute attention only for new tokens
                q = layer['norm1'](x[:, -1:])  # Only last token for query
                k = v = layer['norm1'](x)  # All tokens for key/value
                
                # Concatenate with cached KV if available
                if self.kv_cache[i]['key'].size(1) > 0:
                    k = torch.cat([self.kv_cache[i]['key'], k], dim=1)
                    v = torch.cat([self.kv_cache[i]['value'], v], dim=1)
                
                # Update KV cache
                self.kv_cache[i]['key'] = k
                self.kv_cache[i]['value'] = v
                
                # Compute attention (no need for mask with KV cache)
                attn_output, _ = layer['self_attn'](q, k, v)
                x = x + layer['dropout'](attn_output)
            else:
                # Standard attention for training
                x = layer['norm1'](x)
                attn_output, _ = layer['self_attn'](x, x, x, attn_mask=src_mask)
                x = x + layer['dropout'](attn_output)
            
            # Feed forward
            x = x + layer['dropout'](layer['ff'](layer['norm2'](x)))
        
        # Project back to 24-bit space
        output = self.output_projection(x)
        
        # Apply sigmoid to get values between 0 and 1
        output = torch.sigmoid(output)
        
        return output
    
def generate_square_subsequent_mask(seq_len: int, batch_size: Optional[int] = None, nhead: Optional[int] = None) -> torch.Tensor:
    """Generate causal mask for transformer.
    
    Args:
        seq_len: Length of the sequence
        batch_size: Optional batch size for broadcasting
        nhead: Optional number of attention heads
    Returns:
        mask: Causal mask of shape [batch_size*nhead, seq_len, seq_len] if batch_size and nhead are provided,
             otherwise [seq_len, seq_len]
    """
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    
    if batch_size is not None and nhead is not None:
        # Expand mask for multi-head attention [batch_size*nhead, seq_len, seq_len]
        mask = mask.unsqueeze(0).expand(batch_size * nhead, seq_len, seq_len)
    
    return mask

def train_epoch(model: PixelTransformer,
                data_loader: ImageDataLoader,
                optimizer: optim.Optimizer,
                scheduler: Any,
                criterion: nn.Module,
                clip_grad_norm: float,
                save_dir: str,
                generate_every: int,
                num_samples: int,
                temperature: float,
                global_step: int = 0) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_accuracy = 0
    n_batches = 0
    
    # Get config from model (handle DDP case)
    model_config = model.module.config if hasattr(model, 'module') else model.config
    accumulation_steps = model_config.gradient_accumulation_steps
    
    # Create progress bar
    pbar = tqdm(enumerate(data_loader.train_loader), 
                desc='Training', 
                disable=not accelerator.is_local_main_process)
    
    for batch_idx, batch in pbar:
        # Forward pass with gradient accumulation
        with accelerator.accumulate(model):
            # Create source and target
            src = batch[:, :-1, :]  # All pixels except last
            target = batch[:, 1:, :]  # All pixels except first
            
            # Limit sequence length during training
            if src.size(1) > model_config.train_seq_length:
                # Randomly select a window of train_seq_length tokens
                start_idx = torch.randint(0, src.size(1) - model_config.train_seq_length, (1,)).item()
                end_idx = start_idx + model_config.train_seq_length
                src = src[:, start_idx:end_idx]
                target = target[:, start_idx:end_idx]
            
            # Create mask for the batch
            mask = generate_square_subsequent_mask(
                src.size(1),
                src.size(0),
                model_config.nhead
            ).to(accelerator.device)
            
            # Forward pass
            output = model(src, mask)
            
            # Calculate loss and accuracy
            loss = criterion(output, target)
            accuracy = compute_accuracy(output.detach(), target)
                
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            
            # Backward pass
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), clip_grad_norm)
                
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
                optimizer.zero_grad()
            
            # Increment global step counter
            global_step += 1
            
            # Generate images every N steps
            if global_step % generate_every == 0:
                save_generated_images(
                    model=model,
                    save_dir=save_dir,
                    step=global_step,
                    num_images=num_samples,
                    image_size=model_config.image_size,
                    temperature=temperature,
                    max_seq_length=model_config.inference_seq_length
                )
        
        # Update stats
            total_loss += loss.item() * accumulation_steps
        total_accuracy += accuracy
        n_batches += 1
            
        # Update progress bar
        if accelerator.is_local_main_process:
            current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'acc': f'{accuracy:.2f}%',
                'lr': f'{current_lr:.2e}',
                'seq_len': src.size(1),
                'step': global_step
            })
    
    # Gather metrics across processes
    total_loss = accelerator.gather(torch.tensor(total_loss, device=accelerator.device)).mean().item()
    total_accuracy = accelerator.gather(torch.tensor(total_accuracy, device=accelerator.device)).mean().item()
    
    return {
        'loss': total_loss / n_batches,
        'accuracy': total_accuracy / n_batches,
        'learning_rate': scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr'],
        'global_step': global_step
    }

def validate(model: PixelTransformer,
            data_loader: ImageDataLoader,
            criterion: nn.Module) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    n_batches = 0
    
    # Get config from model (handle DDP case)
    model_config = model.module.config if hasattr(model, 'module') else model.config
    
    with torch.no_grad():
        for batch in data_loader.val_loader:
            # Create source and target
            src = batch[:, :-1, :]
            target = batch[:, 1:, :]
            
            # Create mask
            mask = generate_square_subsequent_mask(
                src.size(1),
                src.size(0),
                model_config.nhead
            ).to(accelerator.device)
            
            # Forward pass
            output = model(src, mask)
            
            # Calculate loss and accuracy
            loss = criterion(output, target)
            accuracy = compute_accuracy(output, target)
            
            # Update stats
            total_loss += loss.item()
            total_accuracy += accuracy
            n_batches += 1
    
    # Gather metrics across processes
    total_loss = accelerator.gather(torch.tensor(total_loss, device=accelerator.device)).mean().item()
    total_accuracy = accelerator.gather(torch.tensor(total_accuracy, device=accelerator.device)).mean().item()
    
    return {
        'loss': total_loss / n_batches,
        'accuracy': total_accuracy / n_batches
    }

def train_model(model: PixelTransformer,
                data_loader: ImageDataLoader,
                optimizer: optim.Optimizer,
                scheduler: Any,
                criterion: nn.Module,
                n_epochs: int,
                save_dir: str = 'checkpoints',
                start_epoch: int = 0,
                initial_history: Optional[Dict] = None,
                save_every: int = 5,
                generate_every: int = 1000,
                num_samples: int = 4,
                temperature: float = 1.0) -> Dict[str, Any]:
    """Main training loop with memory optimization"""
    print("Starting training...")
    os.makedirs(save_dir, exist_ok=True)
    
    # Get config from model (handle DDP case)
    model_config = model.module.config if hasattr(model, 'module') else model.config
    
    # Initialize TensorBoard writer (only on main process)
    writer = None
    if accelerator.is_local_main_process:
        writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))
    
    best_val_loss = float('inf')
    start_time = time.time()
    global_step = 0
    
    # Training history
    history = initial_history or {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rates': []
    }
    
    try:
        for epoch in range(start_epoch, n_epochs):
            # Train and validate
            train_metrics = train_epoch(
                model=model,
                data_loader=data_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                clip_grad_norm=model_config.clip_grad_norm,
                save_dir=save_dir,
                generate_every=generate_every,
                num_samples=num_samples,
                temperature=temperature,
                global_step=global_step
            )
            global_step = train_metrics['global_step']
            
            clear_memory()  # Clear memory after training
            
            val_metrics = validate(model, data_loader, criterion)
            clear_memory()  # Clear memory after validation
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['learning_rates'].append(train_metrics['learning_rate'])
            
            # Log metrics to TensorBoard (only on main process)
            if accelerator.is_local_main_process and writer is not None:
                writer.add_scalar('Loss/train', train_metrics['loss'], global_step)
                writer.add_scalar('Loss/validation', val_metrics['loss'], global_step)
                writer.add_scalar('Accuracy/train', train_metrics['accuracy'], global_step)
                writer.add_scalar('Accuracy/validation', val_metrics['accuracy'], global_step)
                writer.add_scalar('Learning_Rate', train_metrics['learning_rate'], global_step)
            
            # Save checkpoints (only on main process)
            if accelerator.is_local_main_process:
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    checkpoint_path = os.path.join(save_dir, f'best_model.pt')
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        save_path=checkpoint_path,
                        history=history
                    )
                
                if (epoch + 1) % save_every == 0:
                    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        save_path=checkpoint_path,
                        history=history
                    )
            
            # Print progress (only on main process)
            if accelerator.is_local_main_process:
                print(f'Epoch {epoch+1}/{n_epochs} (Step {global_step}):')
                print(f'  Train Loss: {train_metrics["loss"]:.4f}, Accuracy: {train_metrics["accuracy"]:.2f}%')
                print(f'  Val Loss: {val_metrics["loss"]:.4f}, Accuracy: {val_metrics["accuracy"]:.2f}%')
                print(f'  Learning Rate: {train_metrics["learning_rate"]:.2e}')
            
            clear_memory()
            
            # Wait for all processes to sync up
            accelerator.wait_for_everyone()
            
    finally:
        if writer is not None:
            writer.close()
            clear_memory()
    
    return {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'best_val_loss': best_val_loss,
        'training_time': time.time() - start_time,
        'history': history,
        'global_step': global_step
    }

class MultibitBCELoss(nn.Module):
    """Custom loss for multi-bit prediction where each position can have multiple 1s."""
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.bce = nn.BCELoss(reduction=reduction)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities [B, N, 24]
            target: Target bit vectors [B, N, 24]
        Returns:
            loss: Average BCE loss across all bits
        """
        # Compute BCE loss for each bit position independently
        return self.bce(pred, target)

def compute_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute bit-level prediction accuracy.
    Args:
        pred: Predicted probabilities [B, N, 24]
        target: Target bit vectors [B, N, 24]
    Returns:
        accuracy: Percentage of correctly predicted bits
    """
    # Convert predictions to binary (0 or 1)
    pred_bits = (pred >= 0.5).float()
    # Compare with target bits
    correct_bits = (pred_bits == target).float().mean().item()
    return correct_bits * 100

def save_checkpoint(model: PixelTransformer, 
                   optimizer: optim.Optimizer,
                   scheduler: Any,
                   epoch: int,
                   global_step: int,
                   train_metrics: Dict[str, float],
                   val_metrics: Dict[str, float],
                   save_path: str,
                   history: Dict[str, List] = None):
    """Save checkpoint with enhanced information."""
    # Get the underlying model if using DDP
    unwrapped_model = accelerator.unwrap_model(model)
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scheduler_type': type(scheduler).__name__ if scheduler is not None else None,
        'scheduler_config': {
            'warmup_steps': scheduler.warmup_steps if isinstance(scheduler, NoamLRScheduler) else None,
            'factor': scheduler.factor if isinstance(scheduler, NoamLRScheduler) else None,
            'd_model': scheduler.d_model if isinstance(scheduler, NoamLRScheduler) else None,
            'last_lr': scheduler.get_last_lr()[0] if scheduler is not None else None,
            'step': scheduler._step if isinstance(scheduler, NoamLRScheduler) else None
        } if scheduler is not None else None,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'model_config': unwrapped_model.config.to_dict(),
        'model_size': unwrapped_model.get_model_size(),
        'timestamp': datetime.now().isoformat(),
        'history': history,
        'learning_rate': optimizer.param_groups[0]['lr']  # Current learning rate
    }
    
    torch.save(checkpoint, save_path)
    
    # Save a human-readable summary
    summary_path = save_path.replace('.pt', '_summary.json')
    summary = {
        'epoch': epoch,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'model_config': unwrapped_model.config.to_dict(),
        'model_size': unwrapped_model.get_model_size(),
        'timestamp': checkpoint['timestamp'],
        'learning_rate': checkpoint['learning_rate'],
        'scheduler_type': checkpoint['scheduler_type'],
        'scheduler_config': checkpoint['scheduler_config']
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

def find_latest_checkpoint(save_dir: str) -> Optional[str]:
    """Find the latest checkpoint in the save directory."""
    if not os.path.exists(save_dir):
        return None
        
    checkpoints = [f for f in os.listdir(save_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    if not checkpoints:
        return None
        
    # Extract epoch numbers and find the latest
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(save_dir, latest_checkpoint)

def load_checkpoint(path: str,
                   model: PixelTransformer,
                   optimizer: Optional[optim.Optimizer] = None,
                   scheduler: Optional[Any] = None) -> dict:
    """Load checkpoint with enhanced information."""
    # Load checkpoint to CPU first to avoid device mismatch
    checkpoint = torch.load(path, map_location='cpu')
    
    # Move model state dict to the correct device before loading
    model_state = {k: v.to(device) for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(model_state)
    
    if optimizer is not None:
        # Move optimizer state to correct device
        optimizer_state = checkpoint['optimizer_state_dict']
        for state in optimizer_state['state'].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        optimizer.load_state_dict(optimizer_state)
    
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            print(f"Warning: Could not load scheduler state: {e}")
            # Reconstruct scheduler state if possible
            if isinstance(scheduler, NoamLRScheduler):
                scheduler_config = checkpoint.get('scheduler_config', {})
                if scheduler_config:
                    scheduler._step = scheduler_config.get('step', 0)
                    scheduler._rate = scheduler_config.get('last_lr', scheduler._rate)
                    scheduler.warmup_steps = scheduler_config.get('warmup_steps', scheduler.warmup_steps)
                    scheduler.factor = scheduler_config.get('factor', scheduler.factor)
                    scheduler.d_model = scheduler_config.get('d_model', scheduler.d_model)
    
    return checkpoint

def resume_or_start_training(model: PixelTransformer,
                           data_loader: ImageDataLoader,
                           optimizer: optim.Optimizer,
                           scheduler: Any,
                           criterion: nn.Module,
                           n_epochs: int,
                           save_dir: str = 'checkpoints',
                           generate_every: int = 10,
                           num_samples: int = 4,
                           temperature: float = 1.0) -> Dict[str, Any]:
    """Resume training from latest checkpoint or start fresh."""
    latest_checkpoint = find_latest_checkpoint(save_dir)
    start_epoch = 0
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rates': []
    }
    
    if latest_checkpoint:
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        checkpoint = load_checkpoint(latest_checkpoint, model, optimizer, scheduler)
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint.get('history', history)
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("Starting fresh training...")
    
    return train_model(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        n_epochs=n_epochs,
        save_dir=save_dir,
        start_epoch=start_epoch,
        initial_history=history,
        generate_every=generate_every,
        num_samples=num_samples,
        temperature=temperature
    )

class NoamLRScheduler:
    """
    Learning rate scheduler with warmup and decay as described in
    'Attention is All You Need' paper. Learning rate increases linearly
    during warmup and then decays proportionally to the inverse square
    root of the step number.
    """
    def __init__(self, optimizer: optim.Optimizer, d_model: int, warmup_steps: int, factor: float = 1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0
        self._rate = 0
    
    def step(self):
        """Update parameters and learning rate"""
        self._step += 1
        rate = self._get_lr()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        
    def _get_lr(self):
        """Calculate learning rate according to formula"""
        step = self._step
        return self.factor * (self.d_model ** (-0.5) *
                min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))
    
    def get_last_lr(self):
        """Return last computed learning rate"""
        return [self._rate]
    
    def state_dict(self):
        """Return scheduler state for checkpointing"""
        return {
            'step': self._step,
            'rate': self._rate
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint"""
        self._step = state_dict['step']
        self._rate = state_dict['rate']

def generate_image(model: PixelTransformer, 
                 image_size: Tuple[int, int] = (32, 32),
                 temperature: float = 1.0,
                 max_seq_length: int = 64,
                 device = None) -> torch.Tensor:
    """Generate an image autoregressively, pixel by pixel.
    
    Args:
        model: The trained PixelTransformer model
        image_size: Size of image to generate (height, width)
        temperature: Temperature for sampling (higher = more random)
        max_seq_length: Maximum sequence length to keep in memory during generation
        device: Device to generate on
    
    Returns:
        Tensor of shape [3, H, W] containing the generated RGB image
    """
    model.eval()
    
    # Calculate sequence length
    total_seq_len = image_size[0] * image_size[1]
    
    # Start with a random pixel
    current_sequence = torch.rand(1, 1, 24, device=device)  # [batch=1, seq_len=1, bits=24]
    
    # Store all generated pixels
    all_pixels = []
    
    # Clear any existing cache
    model.clear_kv_cache()
    
    # For tracking progress
    last_percentage = -1
    
    with torch.no_grad():
        # Generate pixels one by one
        for i in range(total_seq_len):
            # Print progress every 20%
            current_percentage = (i * 100) // total_seq_len
            if current_percentage % 20 == 0 and current_percentage != last_percentage:
                if accelerator.is_local_main_process:
                    print(f"Image generation: {current_percentage}% complete")
                last_percentage = current_percentage
            
            # Trim sequence if it exceeds max length
            if current_sequence.size(1) > max_seq_length:
                # Keep only the last max_seq_length tokens
                current_sequence = current_sequence[:, -max_seq_length:]
                # Clear KV cache and reinitialize with trimmed sequence
                model.clear_kv_cache()
                
                # Create mask for the trimmed sequence
                mask = generate_square_subsequent_mask(
                    current_sequence.size(1),
                    batch_size=1,
                    nhead=model.config.nhead
                ).to(device)
                
                # Run forward pass on trimmed sequence to rebuild cache
                _ = model(current_sequence, mask, use_cache=True)
            
            # Create mask for the current sequence
            mask = generate_square_subsequent_mask(
                current_sequence.size(1),
                batch_size=1,
                nhead=model.config.nhead
            ).to(device)
            
            # Get model's prediction using key-value caching
            output = model(current_sequence[:, -1:], mask, use_cache=True)
            
            # Get the next pixel prediction
            next_pixel_logits = output[:, -1:, :] / temperature
            
            # Sample from the predicted distribution
            next_pixel_probs = torch.sigmoid(next_pixel_logits)
            next_pixel = (next_pixel_probs > torch.rand_like(next_pixel_probs)).float()
            
            # Store the generated pixel
            all_pixels.append(next_pixel.squeeze(0))
            
            # Append to our sequence
            current_sequence = torch.cat([current_sequence, next_pixel], dim=1)
    
    # Print 100% completion
    if accelerator.is_local_main_process:
        print("Image generation: 100% complete")
    
    # Clear cache after generation
    model.clear_kv_cache()
    
    # Stack all generated pixels
    image_bits = torch.stack(all_pixels, dim=0)  # [total_seq_len, 24]
    
    # Convert bit sequence back to RGB image
    from pixel_bit_encoding import PixelEncoder
    image = PixelEncoder.decode_rgb(image_bits)
    
    # Reshape to proper image dimensions [3, H, W]
    image = image.view(image_size[0], image_size[1], 3)
    image = image.permute(2, 0, 1)  # [C, H, W]
    
    return image

def save_generated_images(model: PixelTransformer,
                       save_dir: str,
                       step: int,
                       num_images: int = 4,
                       image_size: Tuple[int, int] = (32, 32),
                       temperature: float = 1.0,
                       max_seq_length: int = 64):
    """Generate and save sample images.
    
    Args:
        model: The trained model
        save_dir: Directory to save images
        step: Current training step
        num_images: Number of images to generate
        image_size: Size of images to generate
        temperature: Temperature for sampling
        max_seq_length: Maximum sequence length to keep in memory during generation
    """
    import torchvision
    
    # Create save directory
    os.makedirs(os.path.join(save_dir, 'generated_images'), exist_ok=True)
    
    # Generate multiple images with progress bar
    images = []
    pbar = tqdm(range(num_images), 
                desc='Generating images',
                disable=not accelerator.is_local_main_process)
    
    for i in pbar:
        pbar.set_postfix({'image': f'{i+1}/{num_images}'})
        image = generate_image(
            model, 
            image_size, 
            temperature, 
            max_seq_length=max_seq_length,
            device=accelerator.device
        )
        images.append(image)
    
    # Combine into a grid
    image_grid = torchvision.utils.make_grid(images, nrow=2, normalize=True)
    
    # Save the grid
    save_path = os.path.join(save_dir, 'generated_images', f'samples_step_{step:07d}.png')
    torchvision.utils.save_image(image_grid, save_path)
    
    if accelerator.is_local_main_process:
        print(f"\nSaved {num_images} generated images to {save_path}")

def main():
    """Main function to run the training"""
    args = parse_args()
    
    # Initialize accelerator and device first
    accelerator, device = initialize_globals(args)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create data loader with specified image size
    data_loader = ImageDataLoader(
        train_dir="images256x256",
        image_size=(args.image_height, args.image_width),
        batch_size=args.batch_size,
        grayscale=False,  # Use RGB
        linearize=True  # Flatten spatial dimensions
    )
    
    # Create model configuration with specified image size
    model_config = ModelConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        image_size=(args.image_height, args.image_width),  # Use specified image size
        sequence_length=24,  # 24 bits for RGB
        train_seq_length=args.train_seq_length,
        inference_seq_length=args.inference_seq_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        clip_grad_norm=args.clip_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Print image size configuration
    if accelerator.is_local_main_process:
        print(f"\nImage Configuration:")
        print(f"Image Size: {args.image_height}x{args.image_width}")
        print(f"Total Sequence Length: {args.image_height * args.image_width}")
        print(f"Training Sequence Length: {args.train_seq_length}")
        print(f"Inference Sequence Length: {args.inference_seq_length}")
    
    # Create model
    model = PixelTransformer(model_config)
    
    # Get model size before DDP wrapping
    model_size = model.get_model_size()
    
    # Setup optimizer with Transformer-specific parameters
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-7,  # Start with very small learning rate for Noam scheduler
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    # Use Noam scheduler (original Transformer scheduler)
    scheduler = NoamLRScheduler(
        optimizer,
        d_model=model_config.d_model,
        warmup_steps=model_config.warmup_steps,
        factor=2.0
    )
    
    criterion = MultibitBCELoss()
    
    # Prepare for distributed training with accelerate
    model, optimizer, scheduler, data_loader.train_loader, data_loader.val_loader = accelerator.prepare(
        model, optimizer, scheduler, data_loader.train_loader, data_loader.val_loader
    )
    
    # Force an initial scheduler step to set the starting learning rate
    scheduler.step()
    
    # Print training configuration only on main process
    if accelerator.is_local_main_process:
        print("\nTraining Configuration:")
        print(f"Device: {accelerator.device}")
        print(f"Mixed Precision: {args.mixed_precision}")
        print(f"Model Parameters: {model_size:,}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
        print(f"Effective Batch Size: {args.batch_size * args.gradient_accumulation_steps}")
    
    # Train model
    if accelerator.is_local_main_process:
        print("\nStarting training...")
    
    model_results = resume_or_start_training(
                    model=model,
                    data_loader=data_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
        criterion=criterion,
        n_epochs=args.n_epochs,
        save_dir=args.save_dir,
        generate_every=args.generate_every,
        num_samples=args.num_samples,
        temperature=args.temperature
    )
    
    # Print final results only on main process
    if accelerator.is_local_main_process:
        print("\nTraining completed!")
        print(f"Best validation loss: {model_results['best_val_loss']:.4f}")
        print(f"Training time: {model_results['training_time']:.2f} seconds")

if __name__ == '__main__':
    main()
