

import os
import torch
from functools import partial
import torch.nn as nn

from src.data_loaders.data_modules import CIFAR10DataModule
from src.trainers.cnn_trainer import CNNTrainer
from src.models.cnn.metric import TopKAccuracy
from src.models.cnn.vgg11_bn import VGG11_bn
from src.trainers.vgg_trainer import VGGTrainer


os.makedirs("Saved/VGG11BN_pretrained_frozen", exist_ok=True)


q4a_dict = dict(
    name = "VGG11BN_pretrained_frozen",

    # (1) Backbone + head
    model_arch = VGG11_bn,
    model_args = dict(
        layer_config  = [512, 256],      # two FC layers: 512 → 256 → 10
        num_classes   = 10,
        activation    = nn.ReLU,
        norm_layer    = nn.BatchNorm1d,
        fine_tune     = False,           # freeze entire VGG backbone
        weights       = "DEFAULT",       # load ImageNet‐pretrained VGG11_BN
    ),

    # (2) CIFAR-10 DataModule w/ 224×224 + ImageNet normalization
    datamodule = CIFAR10DataModule,
    data_args = dict(
        data_dir         = "data/exercise-2",
        transform_preset = "CIFAR10_VGG",  # must exactly match your preset
        batch_size       = 16,
        shuffle          = True,
        heldout_split    = 0.10,
        num_workers      = 2,
        training         = True,
    ),

    # (3) Optimizer & LR scheduler (only head is trainable)
    optimizer = partial(
        torch.optim.Adam,
        lr = 1e-3,
        weight_decay = 1e-4,
    ),
    lr_scheduler = partial(
        torch.optim.lr_scheduler.StepLR,
        step_size = 7,
        gamma     = 0.1,
    ),

    criterion      = nn.CrossEntropyLoss,
    criterion_args = dict(),

    # (4) Metrics: Top-1 / Top-5
    metrics = dict(
        top1 = TopKAccuracy(k=1),
        top5 = TopKAccuracy(k=5),
    ),

    # (5) Trainer config: no checkpointing, no saving at all
    trainer_module = CNNTrainer,
    trainer_config = dict(
        n_gpu       = 1,
        epochs      = 15,                    # ~12–15 epochs → ~62–65% val Top-1
        eval_period = 1,

        save_dir    = "Saved/VGG11BN_pretrained_frozen",
        save_period = 1,                     # absolutely no checkpoint writes

        monitor     = "max eval_top1",
        early_stop  = 5,                     # early-stop if val_top1 doesn’t improve

        log_step    = 100,
        tensorboard = False,
        wandb       = False,
    ),
)


q4b_finetune_dict = dict(
    name = "VGG11BN_finetune",

    model_arch = VGG11_bn,
    model_args = dict(
        layer_config = [512, 256],      # two FCs: 512 → 256 → 10
        num_classes  = 10,
        activation   = nn.ReLU,         # ReLU between FC/BN
        norm_layer   = nn.BatchNorm1d,  # BatchNorm over the 1D features
        fine_tune    = True,            # <— UNFREEZE all conv‐blocks + head
        weights      = "DEFAULT",       # <— load ImageNet‐pretrained VGG11_BN
    ),

    datamodule = CIFAR10DataModule,
    data_args = dict(
        data_dir         = "data/exercise-2",   # adjust if needed
        transform_preset = "CIFAR10_VGG",       # 224×224 + ImageNet statistics
        batch_size       = 32,                 # you can bump to 128 if GPU allows
        shuffle          = True,
        heldout_split    = 0.10,                # 10% of train as validation
        num_workers      = 4,
        training         = True,
    ),

    optimizer = partial(
        torch.optim.Adam,
        lr = 1e-4,           # smaller LR for fine-tuning (ImageNet → CIFAR domain)
        weight_decay = 1e-4, 
    ),
    lr_scheduler = partial(
        torch.optim.lr_scheduler.StepLR,
        step_size = 7,       # drop LR by 0.1 after epoch 7, etc.
        gamma     = 0.1,
    ),

    criterion      = nn.CrossEntropyLoss,
    criterion_args = dict(),

    # (4) Metrics: Top-1 / Top-5
    metrics = dict(
        top1 = TopKAccuracy(k=1),
        top5 = TopKAccuracy(k=5),
    ),

    # (5) Trainer: VGGTrainer will update entire backbone + head
    trainer_module = VGGTrainer,
    trainer_config = dict(
        n_gpu       = 1,
        epochs      = 20,              # you can train ~15-20 epochs
        eval_period = 1,               # validate every epoch
        save_dir    = "Saved/VGG11BN_finetune",
        save_period = 5,               # save every 5 epochs (optional)
        monitor     = "max eval_top1",
        early_stop  = 5,               # stop if no improvement for 5 evals

        log_step    = 100,
        tensorboard = False,
        wandb       = False,
    ),
)


q4b_scratch_dict = dict(
    name = "VGG11BN_scratch",

    model_arch = VGG11_bn,
    model_args = dict(
        layer_config = [512, 256],      # two FCs: 512 → 256 → 10
        num_classes  = 10,
        activation   = nn.ReLU,
        norm_layer   = nn.BatchNorm1d,
        fine_tune    = True,            # all layers trainable (backbone & head)
        weights      = None,            # <— Random init (no pretrained ImageNet weights)
    ),

    datamodule = CIFAR10DataModule,
    data_args = dict(
        data_dir         = "data/exercise-2",
        transform_preset = "CIFAR10_VGG",   # still use 224×224 + ImageNet stats for consistency
        batch_size       = 32,
        shuffle          = True,
        heldout_split    = 0.10,
        num_workers      = 4,
        training         = True,
    ),

    optimizer = partial(
        torch.optim.Adam,
        lr = 3e-4,            # a bit larger than in fine-tune, but still small
        weight_decay = 1e-4,
    ),
    lr_scheduler = partial(
        torch.optim.lr_scheduler.StepLR,
        step_size = 7,
        gamma     = 0.1,
    ),

    criterion      = nn.CrossEntropyLoss,
    criterion_args = dict(),

    metrics = dict(
        top1 = TopKAccuracy(k=1),
        top5 = TopKAccuracy(k=5),
    ),

    trainer_module = VGGTrainer,
    trainer_config = dict(
        n_gpu       = 1,
        epochs      = 30,              # training from scratch may need ~25-30 epochs
        eval_period = 1,
        save_dir    = "Saved/VGG11BN_scratch",
        save_period = 5,
        monitor     = "max eval_top1",
        early_stop  = 7,               # a bit more patience for scratch

        log_step    = 100,
        tensorboard = False,
        wandb       = False,
    ),
)
