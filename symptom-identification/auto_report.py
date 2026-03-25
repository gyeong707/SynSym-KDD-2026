import pandas as pd
import argparse
import json
from parse_config import ConfigParser


def save_report(result, dir_path, eval_type):
    print("Save the result to ", dir_path)
    file_name = f"/classification_report_"+eval_type+"_transformed.csv"
    result.to_csv(dir_path+file_name, index=True)
    print("Save Report.")
    
def label_report(label, df):
    print("Target Label: ", label)
    value = df[df.index == label][['precision', 'recall', 'f1-score', 'support']]
    return value.reset_index(drop=True)

def auto_report(dir_path, label_name, eval_type):
    # load data
    print("Load the result from ", dir_path)
    report_path = dir_path + "/classification_report_"+eval_type+".csv"
    df = pd.read_csv(report_path, index_col=0)

    avg_name = ["micro avg", "macro avg", "weighted avg", "samples avg"]   
    label_name = avg_name + label_name
    # create summary for each label
    label_results = []
    for label in label_name:
        label_metrics = label_report(label, df)
        print(label_metrics)
        label_results.append(label_metrics)
    
    # combine all results
    result = pd.concat(label_results, axis=1)
    save_report(result, dir_path, eval_type)
    

# main function
if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Auto report for result of symptom identification.")
    args.add_argument('-d', '--device', default=None, type=str, help='device')
    args.add_argument("-r", "--resume", default=None, type=str, help="resume")
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument("--dir_path", type=str, required=True, help="Directory of the input/output file")
    args.add_argument("--eval", type=str, required=True, help="Type of evaluation: test or val")

    # config
    config = ConfigParser.from_args(args)

    args = args.parse_args()
    dir_path = f"./saved/log/{args.dir_path}"

    if "label_dictionary" in config['data_loader']['dict_dir']:
        print(":: Auto Report for NLI ::")
        with open(config['data_loader']['load_dir']+"symptom_dictionary.json", 'r') as f:
            label_dict = json.load(f)
    else:
        with open(config['data_loader']['dict_dir'], 'r') as f:
            label_dict = json.load(f)
    label_name = list(label_dict.keys()) 

    auto_report(dir_path, label_name, args.eval)
    print("Done.")