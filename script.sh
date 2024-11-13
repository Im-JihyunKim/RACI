#!/bin/bash

for seed in 0 1 2 4 42 418
do
    for val_ratio in .1 .2 .3 .4 .5
    do
        for backbone in CNN DNN Transformer
        do
            for et_weight in var range same
            do
                echo $seed, val:$val_ratio, $backbone, $et_weight
                
                python main.py --backbone $backbone \
                                --et-weight $et_weight \
                                --seed=$seed \
                                --valid-ratio $val_ratio \
                                --task ET \
                                --batch-size 32 \
                                --epochs 1000 \
                                --patience 500 \
                                --encoding-type Count \
                                --eqp eqp \
                                --NP NP \
                                --lr 5e-3 \
                                --num-workers 0
            done
        done
    done
done