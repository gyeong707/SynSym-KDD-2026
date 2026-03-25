#!/bin/bash

SESSION_NAME="MultiLabelClassification"
MODEL_NAME=("mentalbert") 
DATA_NAME=("Psysym")
OPTION_NAME_1="dsm5"
OPTION_NAME_2="with_real" 
CKP_FOLD=(0)
FOLD=(0)
SEED=(42)

for MODEL in "${MODEL_NAME[@]}"; do
    for DATA in "${DATA_NAME[@]}"; do
        for CKP_F in "${CKP_FOLD[@]}"; do
            for S in "${SEED[@]}"; do
                for F in "${FOLD[@]}"; do
                    MODEL_PATH=$SESSION_NAME/$MODEL/Synth/$OPTION_NAME_1/default/seed_$S/fold_$CKP_F
                    CONFIG_PATH="config/Synth/depression/$DATA/$OPTION_NAME_2/config_mlc_${MODEL}_${OPTION_NAME_2}_fold_${CKP_F}.json"
                    echo "Running training script with $MODEL_PATH with seed $S, ckp_fold $CKP_F, and fold $F"
                    python test_psysym.py -c $CONFIG_PATH -ckp "saved/models/$MODEL_PATH/model_best.pth" -eval "test" -seed "$S" -fold "$F"
                    python auto_report.py --dir_path "$SESSION_NAME/$MODEL/Synth/$DATA/depression/$OPTION_NAME_2/ckp_fold_$CKP_F/seed_$S/fold_$F" -c $CONFIG_PATH --eval "test"
                    echo "Finished training script with $MODEL_PATH"
                done
            done
            python auto_merge_seeds_psysym.py --dir_path "saved/log/$SESSION_NAME/$MODEL/Synth/$DATA/depression/$OPTION_NAME_2/ckp_fold_$CKP_F"
        done
    done
done


