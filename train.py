import os
import json
import yaml
import random
import argparse
import datetime
import numpy as np
import pickle

import torch

from data.dataloader import select_dataloader
from utils.log import print_log, CustomJSONEncoder
from utils.metrics import select_loss
from runners.FCoSDLTSFRunner import FCoSDLTSFRunner
from model.FCoSDNet import FCoSD

parser = argparse.ArgumentParser()
# Specify the dataset
parser.add_argument('-d','--dataset_name',type=str, default='ENERGY')
# Specify config file path
parser.add_argument('-cfg','--config_path',type=str, default='./config/ENERGY/ENERGY_Seq96.yaml')
# Specify random seed
parser.add_argument('-sd','--seed',type=int, default=2025)
# checkpoint path for testing
parser.add_argument('-c','--checkpoint',type=str, default='./checkpoints/ENERGY/ENERGY-2026-01-22-17-46-59-best1.pt')

args = parser.parse_args()

os.environ['PYTHONHASHSEED'] = str(args.seed)

random.seed(args.seed)
np.random.seed(args.seed)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def load_checkpoint(model, checkpoint_path, device):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        # If the checkpoint is just the model state dict
        model.load_state_dict(checkpoint, strict=False)
    
    print("Checkpoint loaded successfully")
    return model

def test_only(model, test_loader, laplacian, runner, device, log):
    print_log("Testing pre-trained model from checkpoint...", log=log)
    runner.test_model(model, test_loader, laplacian)

def train():
    # Set device
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(DEVICE)

    cfg_path = args.config_path
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    data_path = f'./data/processed_data/{args.dataset_name}/'
    data_path_lap = f'./data/processed_data/{args.dataset_name}'
    mode = cfg['GENERAL'].get('mode', 'train')

    # --------- Load the model --------- #
    model = FCoSD(**cfg['MODEL_PARAM']).to(DEVICE)

    # --------- Make log file --------- #
    log_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    log_path = f'./logs/{args.dataset_name.upper()}'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f"{args.dataset_name.upper()}-init-{cfg['MODEL_PARAM'].get('num_bands', 3)}-{cfg['MODEL_PARAM'].get('rank', 20)}-{cfg['MODEL_PARAM'].get('op_w', 0)}-{log_time}-{mode}.log")
    log = open(log, 'a')
    log.seek(0)
    log.truncate()

    # --------- Load the dataset --------- #
    print_log(f'Dataset used: {args.dataset_name.upper()}', log=log)
    (train_loader, val_loader, test_loader, SCALER) = select_dataloader(cfg["GENERAL"]["task_name"])(
        data_path,
        batch_size=cfg['GENERAL'].get('batch_size', 32),
        in_steps=cfg['DATA'].get('in_steps', 96),
        out_steps=cfg['DATA'].get('out_steps', 96),
        x_tod=cfg['DATA'].get('x_time_of_day'),
        x_dow=cfg['DATA'].get('x_day_of_week'),
        y_tod=cfg['DATA'].get('y_time_of_day'),
        y_dow=cfg['DATA'].get('y_day_of_week'),
        log=log
    )
    print_log(log=log)

    # --------- Set checkpoint path --------- #
    save_path = f'./checkpoints/{args.dataset_name.upper()}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f"{args.dataset_name.upper()}-{cfg['MODEL_PARAM'].get('num_bands', 3)}-{cfg['MODEL_PARAM'].get('rank', 20)}-{cfg['MODEL_PARAM'].get('op_w', 0)}-{log_time}.pt")

    # --------- Set optim options --------- #
    criterion = select_loss(cfg['OPTIM'].get('loss', 'MSE'))(**cfg['OPTIM'].get('loss_args', {}))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg['OPTIM'].get('initial_lr', 0.001),
        weight_decay=cfg['OPTIM'].get('weight_decay', 0),
        eps=cfg['OPTIM'].get('eps', 1e-8)
    )

    lr_scheduler_type = cfg['OPTIM'].get('lr_scheduler_type', 'ExponentialLR')
    if lr_scheduler_type == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg['OPTIM'].get('lr_scheduler_gamma', 0.5),
            verbose=False
        )
    elif lr_scheduler_type == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            steps_per_epoch=len(train_loader),
            max_lr= cfg['OPTIM'].get('initial_lr'),
            epochs=cfg['GENERAL'].get('max_epochs'),
            pct_start=cfg['OPTIM'].get('lr_scheduler_pct_start')
        )
    elif lr_scheduler_type == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg.get('milestones', []),
        gamma=cfg.get('lr_decay_rate', 0.1),
        verbose=False
        )
    else: 
        raise ValueError('No such lr scheduler') 

    # --------- Set model runner --------- #

    runner = FCoSDLTSFRunner(
        cfg, device=DEVICE, scaler=SCALER, log=log)

    # --------- Print model args --------- #
    print_log(f'Random seed = {args.seed}', log=log)
    print_log(
        json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log
    )

    # --------- Train the model --------- #
    print_log(f'Model checkpoint saved to: {save}', log=log)
    print_log(log=log)

    with open(f'{data_path_lap}/adj_{args.dataset_name}.pkl','rb') as f:
        laplacian = pickle.load(f)

    laplacian = np.array(laplacian)
    laplacian = torch.tensor(laplacian, dtype=torch.float32).to(DEVICE)

    # --------- Load the model --------- #
    if mode == 'test':
        model = FCoSD(**cfg['MODEL_PARAM']).to(DEVICE)

        if args.checkpoint is not None:
            # Load model from checkpoint
            model = load_checkpoint(model, args.checkpoint, DEVICE)
            
            # Test the model directly
            test_only(model, test_loader, laplacian, runner, DEVICE, log)
            
            log.close()
            torch.cuda.empty_cache()
            return

    model = runner.train(
        model,
        train_loader,
        val_loader,
        laplacian,
        optimizer,
        scheduler,
        criterion,
        max_epochs=cfg['GENERAL'].get('max_epochs', 10),
        early_stop_patience=cfg['GENERAL'].get('early_stop_patience', 3),
        compile_model=False,
        verbose=1,
        save=save,
    )

    # --------- Test the model --------- #
    runner.test_model(model, test_loader, laplacian)

    log.close()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    train()
