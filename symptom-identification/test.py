import argparse
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from model.tokenizer import load_tokenizer
from torch.utils.data import Dataset, DataLoader
from model.metric import print_classification_report
from sklearn.metrics import classification_report
import json
import collections
import random
import os
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef


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
    logger = config.get_logger('test')
    eval_type = config["eval"]

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
    threshold = config["metrics"]["threshold"]
    fold = config["info"]["fold"] 

    # symptom labels
    label_index = module_data.make_label_vector(data, label_cols)
    config["metrics"]["target_name"] = list(label_dict.keys()) 
    target_name = config["metrics"]["target_name"]
    print("Target Name : ", list(label_dict.keys()), ">>", len(list(label_dict.keys())))

    # Multi-label straitified split
    X_train_, X_test, y_train_, y_test, train_index, test_index = module_data.multilabel_stratified_split(
        data, label_index, SEED, fold, n_splits=5, shuffle=True, column_name=text_cols)

    test_dataset = module_data.MLCDataset(X_test, y_test, tokenizer, max_length)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

    # build model architecture
    if config["n_gpu"] == 1:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    elif config["n_gpu"] == 0:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = config.init_obj("arch", module_arch)


    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']["calculator"]]

    logger.info('Loading checkpoint: {} ...'.format(config["resume"]))
    checkpoint = torch.load(config["resume"], weights_only=False)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    with torch.no_grad():
        all_labels = np.array([])
        all_preds = np.array([])
        all_logits = []

        for batch_idx, batch in enumerate(tqdm(test_dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            output = model(input_ids, attention_mask)

            # adding values
            output = torch.sigmoid(output)
            output_np = (output.detach() > threshold).cpu().numpy()
            labels_np = labels.cpu().numpy()

            if batch_idx == 0:
                all_labels = labels_np
                all_preds = output_np
            else:
                all_labels = np.vstack((all_labels, labels_np))
                all_preds = np.vstack((all_preds, output_np))
            
            # logits for error anlaysis
            all_logits.extend(output.cpu().numpy())

            # computing loss, metrics on test set
            loss = loss_fn(output, labels)
            total_loss += loss.item()

    print("all_preds : ", all_preds.shape, all_preds[0])
    print("all_labels : ", all_labels.shape, all_labels[0])
    
    # total_loss
    n_samples = len(test_dataloader.sampler)
    log = {'loss': total_loss / n_samples}

    # metric
    met_log = {}
    for met in metric_fns:
        met_log[met.__name__] = met(all_labels, all_preds)
    log.update(**{"val_" + k: v for k, v in met_log.items()})
    logger.info(log)
    
    # classification_report
    report = (
    print_classification_report(
        target_name, all_labels, all_preds, output_dict=True
    ))
    # report_lines = report.split('\n')
    # for line in report_lines:
    #     logger.info(line)
        
    report_str = json.dumps(report, indent=2)
    report_lines = report_str.split('\n')
    for line in report_lines:
        logger.info(line)
        
    df = pd.DataFrame(report).transpose()
    df.to_csv(str(config.log_dir)+"/classification_report_"+eval_type+".csv")

    ## error report
    text = X_test # testing
    pred = module_data.revert_label_index(all_preds, label_dict)
    label = module_data.revert_label_index(all_labels, label_dict)
    logits = all_logits
    res_df = pd.DataFrame(data={'text': text, 'label': label, 'pred': pred, 'logits': logits})
    res_df.to_csv(str(config.log_dir)+'/error_analysis_'+eval_type+'.csv', index=False)


    mccs = []
    for label in range(config['arch']['args']['num_labels']):
        mcc = matthews_corrcoef(all_labels[:, label], all_preds[:, label])
        mccs.append(mcc)
        logger.info(f"Label {label} MCC: {mcc:.4f}")

    mcc_df = pd.DataFrame({
        'label': target_name,
        'mcc': mccs
    })
    mcc_df.to_csv(str(config.log_dir)+f"/mcc_scores_{eval_type}.csv", index=False)
        

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument("-r", "--resume", default=None, type=str, 
                      help="path to latest checkpoint (default: None)")
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["-lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(["-bs", "--batch_size"], type=int, target="data_loader;args;batch_size"),
        CustomArgs(["-ckp", "--ckp"], type=str, target="resume"),
        CustomArgs(["-opt", "--option"], type=str, target="info;option"),
        CustomArgs(["-data", "--data"], type=int, target="info;data"),
        CustomArgs(["-eval", "--eval"], type=str, target="eval"),
        CustomArgs(["-fold", "--fold"], type=int, target="info;fold"),
        CustomArgs(["-seed", "--seed"], type=int, target="info;seed"),
        CustomArgs(["--mode", "--mode"], type=str, target="info;mode"),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
