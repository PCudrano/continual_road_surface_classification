#!/usr/bin/env python
# coding: utf-8

# import cv2
# import csv
import os
import numpy as np
from collections import Counter, OrderedDict
import torch
# from torchvision.transforms import ToTensor
# from torch.utils.data import DataLoader
from torchsummary import summary
import torchvision
# import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks import nc_benchmark, ni_benchmark, dataset_benchmark
from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (forgetting_metrics, accuracy_metrics, loss_metrics, cpu_usage_metrics,
                                          timing_metrics, gpu_usage_metrics, ram_usage_metrics, disk_usage_metrics,
                                          MAC_metrics, confusion_matrix_metrics, amca_metrics,
                                          forward_transfer_metrics, bwt_metrics)
from avalanche.training.supervised import Naive, Cumulative, JointTraining
from avalanche.training import LFL
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins import EarlyStoppingPlugin

from includes.dataloaders.load_dataloaders import load_dataloader_rtk_paper
from includes.models.rtk_cnn import RtkModel
from includes.models.custom_resnet import CustomResNet


# Configs

torch.manual_seed(42)

BASE = '/home/cudrano'

LOOP = False  # False | int

wandb_args = dict(
    project="continual_road_surface_baseline",
    notes="LFL new eval", # "Baseline", # "Baseline loop",
    #notes="Baseline cumulative",
    #notes="Baseline joint",
    #notes="Baseline naive",
    tags=["lfl", "rtk_paper"], # "cl_baseline",
    #tags=['cl_baseline_cumul', 'rtk_paper']
    #tags=['cl_baseline_joint', 'rtk_paper']
    #tags=['cl_baseline', "rtk_paper"]
)
hyperpar = dict(
    run="cl_lfl", # "cl_baseline2", # "cl_baseline_loop"
    #run='cl_cumulative',
    #run='cl_joint',
    #run="cl_naive",
    strategy='lfl', 
    #strategy='cumulative',
    #strategy='joint',
    #strategy='naive', #'lfl',
    dataset_id="rtk_kitti_carina",
    dataset_ext='enlarged',  # 'compact',
    architecture="rtk_model",  # "resnet18",
    optimizer="SGD",  # "Adam",
    #optimizer="Adam",
    momentum=0.9,
    weight_decay=1e-8,#1e-8,  # 5e-4,
    learning_rate=0.002,
    lambda_e=0.5,
    early_stopping='none',  #
    epochs=10,  # 150
    batch_size=32,
    cuda=0,
    load_ds_on_device=False,
    use_class_weights=True  # False
)

# Init constants

BASE_DATASET_PATH = os.path.join(BASE, 'data')
RTK_DATASET_PATH = os.path.join(BASE_DATASET_PATH, f'{hyperpar["dataset_ext"]}_dataset_RTK')
KITTI_DATASET_PATH = os.path.join(BASE_DATASET_PATH, f'{hyperpar["dataset_ext"]}_dataset_KITTI')
CARINA_DATASET_PATH = os.path.join(BASE_DATASET_PATH, f'{hyperpar["dataset_ext"]}_dataset_CaRINA')

WANDB_PATH = os.path.join(BASE, 'outputs/wandb')
CKPT_PATH = os.path.join(BASE, 'outputs/ckpts')
TB_PATH = os.path.join(BASE, 'outputs/tb_data')
if not os.path.exists(WANDB_PATH):
    os.mkdir(WANDB_PATH)
if not os.path.exists(CKPT_PATH):
    os.mkdir(CKPT_PATH)
if not os.path.exists(TB_PATH):
    os.mkdir(TB_PATH)

LABELS = {'asphalt': 0,
          'paved': 1,
          'unpaved': 2}
NUM_CLASSES = len(LABELS.keys())
#IMG_H = 288
#IMG_W = 352
CROPPED_H = 128#144
CROPPED_W = 352
NUM_CHANNELS = 3

TRAIN_SPLIT = 0.6 if hyperpar['early_stopping'] and hyperpar['early_stopping'] != 'none' else 0.8
VALID_SPLIT = 0.2 if hyperpar['early_stopping'] and hyperpar['early_stopping'] != 'none' else 0
TEST_SPLIT = 1 - VALID_SPLIT - TRAIN_SPLIT

rtk_cropping = [-1, 0, 0.17, 0] #0.0 # 0.17
kitti_cropping = [-1, 0, 0.17, 0] #0.0
carina_cropping = [-1, 0, 0.17, 0] #0.11

device = torch.device(
        f"cuda:{hyperpar['cuda']}"
        if torch.cuda.is_available() and hyperpar['cuda'] >= 0
        else "cpu"
    )

# Load Dataset

kwargs = {'num_workers': 0, 'pin_memory': True} if device.type == 'cuda' else {}
to_device = device if device.type == 'cuda' and hyperpar['load_ds_on_device'] else None

rtk_train_dataloader, rtk_valid_dataloader, rtk_test_dataloader, rtk_train_ds, rtk_valid_ds, rtk_test_ds = load_dataloader_rtk_paper(
        RTK_DATASET_PATH, LABELS, rtk_cropping, [CROPPED_H, CROPPED_W],
        TRAIN_SPLIT, VALID_SPLIT, hyperpar['batch_size'], to_device=to_device, **kwargs)
print("Loaded RTK: ", rtk_train_dataloader.batch_size*len(rtk_train_dataloader), rtk_valid_dataloader.batch_size*len(rtk_valid_dataloader), rtk_test_dataloader.batch_size*len(rtk_test_dataloader))
kitti_train_dataloader, kitti_valid_dataloader, kitti_test_dataloader,     kitti_train_ds, kitti_valid_ds, kitti_test_ds = load_dataloader_rtk_paper(
        KITTI_DATASET_PATH, LABELS, kitti_cropping, [CROPPED_H, CROPPED_W],
        TRAIN_SPLIT, VALID_SPLIT, hyperpar['batch_size'], to_device=to_device, **kwargs)
print("Loaded KITTI: ", kitti_train_dataloader.batch_size*len(kitti_train_dataloader), kitti_valid_dataloader.batch_size*len(kitti_valid_dataloader), kitti_test_dataloader.batch_size*len(kitti_test_dataloader))
carina_train_dataloader, carina_valid_dataloader, carina_test_dataloader,     carina_train_ds, carina_valid_ds, carina_test_ds = load_dataloader_rtk_paper(
        CARINA_DATASET_PATH, LABELS, carina_cropping, [CROPPED_H, CROPPED_W],
        TRAIN_SPLIT, VALID_SPLIT, hyperpar['batch_size'], to_device=to_device, **kwargs)
print("Loaded CaRINA: ", carina_train_dataloader.batch_size*len(carina_train_dataloader), carina_valid_dataloader.batch_size*len(carina_valid_dataloader), carina_test_dataloader.batch_size*len(carina_test_dataloader))

# Class weights
n_samples_per_class = torch.tensor(list(OrderedDict(sorted(dict(Counter(rtk_train_ds.targets)).items())).values()))
class_weights = (torch.sum(n_samples_per_class) / n_samples_per_class).to(device)
print("Class samples: {}, Class weights: {}".format(n_samples_per_class, class_weights))

train_ds_list = [rtk_train_ds, kitti_train_ds, carina_train_ds]
valid_ds_list= [rtk_valid_ds, kitti_valid_ds, carina_valid_ds]
test_ds_list = [rtk_test_ds, kitti_test_ds, carina_test_ds]
ds_order = [0,1,2]
if LOOP and LOOP > 0:
    ds_order = np.tile(ds_order,LOOP)
print("Dataset order: ", ds_order)

# Scenario

scenario = dataset_benchmark(
        [train_ds_list[i] for i in ds_order],
        [test_ds_list[i] for i in ds_order],
        other_streams_datasets=
            {'valid': [valid_ds_list[i] for i in ds_order]} if VALID_SPLIT > 0
            else {'valid': [test_ds_list[i] for i in ds_order]}
    )

# Define model

if hyperpar['architecture'] == 'rtk_model':
    model = RtkModel()
elif hyperpar['architecture'] == 'resnet18':
    model = CustomResNet(torchvision.models.resnet18())
model = model.to(device)
summary(model, (NUM_CHANNELS, CROPPED_H, CROPPED_W))

# Loggers

interactive_logger = InteractiveLogger()
wandb_logger = WandBLogger(
    project_name=wandb_args['project'],
    run_name=hyperpar['run'],
    path=CKPT_PATH,  # checkpoints path
    save_code=True, 
    dir=WANDB_PATH,  # wandb data path
    log_artifacts=True,
    params=wandb_args,  # params passed to wandb.init
    config=hyperpar  # hyperparameters
)
# tb_logger = TensorboardLogger(tb_log_dir=TB_PATH)

# Evaluation plugin
eval_plugin = EvaluationPlugin(
    accuracy_metrics(
        minibatch=True,
        epoch=True,
        epoch_running=True,
        experience=True,
        stream=True,
    ),
    loss_metrics(
        minibatch=True,
        epoch=True,
        epoch_running=True,
        experience=True,
        stream=True,
    ),
    amca_metrics(),
    forgetting_metrics(experience=True, stream=True),
    bwt_metrics(experience=True, stream=True),
    #forward_transfer_metrics(experience=True, stream=True),
    confusion_matrix_metrics(
        stream=True, wandb=True, num_classes=NUM_CLASSES, class_names=list(LABELS.keys())  # [str(i) for i in range(10)]
    ),
#   cpu_usage_metrics(
#       minibatch=True, epoch=True, experience=True, stream=True
#   ),
#   timing_metrics(
#       minibatch=True, epoch=True, experience=True, stream=True
#   ),
#   ram_usage_metrics(
#       every=0.5, minibatch=True, epoch=True, experience=True, stream=True
#    ),
#    gpu_usage_metrics(
#        hyperpar['cuda'],
#       every=0.5,
#       minibatch=True,
#       epoch=True,
#       experience=True,
#       stream=True,
#   ),
#    disk_usage_metrics(
#        minibatch=True, epoch=True, experience=True, stream=True
#    ),
    MAC_metrics(minibatch=True, epoch=True, experience=True),
    # loggers=[interactive_logger, wandb_logger, tb_logger],
    loggers=[interactive_logger, wandb_logger],
    # collect_all=True
)

# Define strategy

if hyperpar['optimizer'] and hyperpar['optimizer'] == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=hyperpar['learning_rate'],
                                momentum=hyperpar['momentum'], weight_decay=hyperpar['weight_decay'])
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperpar['learning_rate'])
if hyperpar['use_class_weights']:
    loss = CrossEntropyLoss(weight=class_weights)
else:
    loss = CrossEntropyLoss()
if hyperpar['early_stopping'] and hyperpar['early_stopping'] != 'none':
    plugins = [EarlyStoppingPlugin(patience=2, peval_mode='epoch',
                                   val_stream_name='valid_stream', metric_name='Top1_Acc_Exp')]
else:
    plugins = []
strategy_kwargs = {
    "plugins": plugins,
    "train_mb_size": hyperpar['batch_size'],
    "train_epochs": hyperpar['epochs'],
    # eval_mb_size=args['batch_size'],
    "eval_every": 1,  # eval every n $peval_mode
    #"peval_mode": 'epoch',  # 'epoch'|'iteration'
    "device": device,
    "evaluator": eval_plugin
}
if hyperpar['strategy'] == 'naive':
    strategy_class = Naive
if hyperpar['strategy'] == 'cumulative':
    strategy_class = Cumulative
if hyperpar['strategy'] == 'joint':
    strategy_class = JointTraining
elif hyperpar['strategy'] == 'lfl':
    strategy_class = LFL
    strategy_kwargs.update({
        'lambda_e': hyperpar['lambda_e']
    })
cl_strategy = strategy_class(
    model=model,
    optimizer=optimizer,
    criterion=loss,
    **strategy_kwargs)

# Training

print("Starting experiment...")

results = []
for i, experience in enumerate(scenario.train_stream):
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    cl_strategy.train(experience) #, eval_streams=[scenario.valid_stream[i:(i+1)]])
    print("Training completed")

    print("Computing accuracy on the whole test set")
    results.append(cl_strategy.eval(scenario.test_stream))

print(f"Test metrics:\n{results}")

# Dict with all the metric curves,
# only available when `collect_all` is True.
# Each entry is a (x, metric value) tuple.
# You can use this dictionary to manipulate the
# metrics without avalanche.
all_metrics = cl_strategy.evaluator.get_all_metrics()

