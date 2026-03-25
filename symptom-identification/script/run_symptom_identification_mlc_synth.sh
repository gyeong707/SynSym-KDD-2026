#!/bin/bash

SESSION_NAME="MultiLabelClassification"
MODEL_NAME=("mentalbert")
DATA_NAME=("Synth")
OPTION_NAME_1="dsm5"
OPTION_NAME_2="default"
SEED=(42)
CKP_FOLD=(0)


for MODEL in "${MODEL_NAME[@]}"; do
    for DATA in "${DATA_NAME[@]}"; do
        for CKP_F in "${CKP_FOLD[@]}"; do
            for S in "${SEED[@]}"; do
                MODEL_PATH=$SESSION_NAME/$MODEL/$DATA/$OPTION_NAME_1/$OPTION_NAME_2/fold_$CKP_F
                CONFIG_PATH="config/$DATA/depression/$OPTION_NAME_1/$OPTION_NAME_2/config_mlc_${MODEL}.json"
                echo "Running training script with $MODEL_PATH"
                python train.py -c $CONFIG_PATH -fold "$CKP_F" -mode "pretrain" -seed "$S" 
                echo "Finished training script with $MODEL_PATH"
            done
        done
    done
done
