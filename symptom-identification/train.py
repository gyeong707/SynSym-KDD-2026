import argparse
import collections
import torch
import scipy
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from model.tokenizer import load_tokenizer
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import math
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ExponentialLR
import json
import random
import os
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# fix random seeds for reproducibility
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def main(config):
    SEED = config['info']['seed']
    seed_everything(SEED)
    logger = config.get_logger("train")

    # load dataset and tokenizer
    tokenizer = load_tokenizer(config["arch"]["args"]["model_type"])
    data = pd.read_csv(config["data_loader"]["data_dir"])
    with open(config['data_loader']['dict_dir'], 'r') as f:
        label_dict = json.load(f)    
        config['arch']['args']['num_labels'] = len(label_dict)

    # arguments
    max_length = config["data_loader"]["max_length"]
    label_cols = config["data_loader"]["label_cols"]
    text_cols = config["data_loader"]["text_cols"]
    batch_size = config["data_loader"]["batch_size"]
    fold = config["info"]["fold"] 

    # symptom labels
    label_index = module_data.make_label_vector(data, label_cols)
    config["metrics"]["target_name"] = list(label_dict.keys())
    print("Target Name : ", list(label_dict.keys()), ">>", len(list(label_dict.keys())))

    # Multi-label straitified split
    X_train, X_test, y_train, y_test, train_index, test_index = module_data.multilabel_stratified_split(
        data, label_index, SEED, fold, n_splits=5, shuffle=True, column_name=text_cols)

    print("Train size: ", len(X_train), "Valid size: ", len(X_test))

    # Load dataset
    train_dataset = module_data.MLCDataset(X_train, y_train, tokenizer, max_length)
    valid_dataset = module_data.MLCDataset(X_test, y_test, tokenizer, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False)

    # build model architecture, then print to console
    device, device_ids = prepare_device(config["n_gpu"])
    model = config.init_obj("arch", module_arch)
    print("Device: ", device)

    if config["resume"] != "":
        print("Loading checkpoint: {} ...".format(config["resume"]))
        logger.info('Loading checkpoint: {} ...'.format(config["resume"]))
        checkpoint = torch.load(config["resume"], weights_only=False)
        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)
        model = model.to(device)    
    else: 
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device)    

    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config["loss"])
    metric_fns = [getattr(module_metric, met) for met in config["metrics"]["calculator"]]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())   
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params)

    # set parameters for linear scheduler with warmup
    if config['lr_scheduler']['use']:
        warmup_proportion = config["lr_scheduler"]["warmup_proportion"]
        num_training_steps = len(train_dataloader) * config["trainer"]["epochs"]
        num_warmup_steps = math.ceil(num_training_steps * warmup_proportion)

        # define learning rate scheduler
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        print(num_training_steps, num_warmup_steps)
    else: 
        lr_scheduler = None

    # define Trainer instance
    trainer = Trainer(
        model,
        criterion,
        metric_fns,
        optimizer,
        config=config,
        device=device,
        data_loader=train_dataloader,
        valid_data_loader=valid_dataloader,
        lr_scheduler=lr_scheduler
    )

    # run training
    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument("-r", "--resume", default=None, type=str, 
                      help="path to latest checkpoint (default: None)")
    
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"),
        CustomArgs(["-ckp", "--ckp"], type=str, target="resume"),
        CustomArgs(["-data", "--data"], type=int, target="info;data"),
        CustomArgs(["-opt", "--option"], type=str, target="info;option"),
        CustomArgs(["-fold", "--fold"], type=int, target="info;fold"),
        CustomArgs(["-seed", "--seed"], type=int, target="info;seed"),
        CustomArgs(["-mode", "--mode"], type=str, target="info;mode"),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
