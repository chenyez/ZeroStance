# nohup bash ./train_ROBERTA_large_cross_dataset_seed0_tune1_all_datasets.sh > train_ROBERTA_large_seed0_cross_datasets.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0
###################################################################################################################
###################################################################################################################
###################################################################################################################
# # Each human-annotated dataset as open-domain dataset, e.g., vast
# config=../config/config-roberta_large.txt
# train_data=../data/
# dev_data=../data/
# test_data=../data/
# all_datasets=("vast" "ibm30k" "covid19" "semeval2016" "wtwt" "pstance")
# for seed in 4 5 7
# do
#     echo "Start training on seed ${seed}......"
#     for dataset in "chatgpt_vast"
#     do
#         echo "Start training on dataset ${dataset}......"
#         python train_model_v2.py ${all_datasets[@]} --leave_one_out 0 -lr1 1e-5 -lr2 1e-5 -d 0.1 -s ${seed} -clipgrad True -step 3 --earlystopping_step 5 -dataset ${dataset} -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data}
#     done
# done

# # ZeroStance w/o data filtering
# config=../config/config-roberta_large.txt
# train_data=../data/
# dev_data=../data/
# test_data=../data/
# all_datasets=("vast" "ibm30k" "covid19" "semeval2016" "wtwt" "pstance")
# for seed in 4 5 7
# do
#     echo "Start training on seed ${seed}......"
#     for dataset in "chatgpt"
#     do
#         echo "Start training on dataset ${dataset}......"
#         python train_model_v2.py ${all_datasets[@]} --leave_one_out 0 -lr1 1e-5 -lr2 1e-5 -d 0.1 -s ${seed} -clipgrad True -step 3 --earlystopping_step 5 -dataset ${dataset} -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data}
#     done
# done

# ZeroStance with data filtering, complete model
config=../config/config-roberta_large.txt
train_data=../data/
dev_data=../data/
test_data=../data/
all_datasets=("vast" "ibm30k" "covid19" "semeval2016" "wtwt" "pstance")
for seed in 4 5 7
do
    echo "Start training on seed ${seed}......"
    for dataset in "chatgpt_carto_bertweet_var_0.99_seed0"
    do
        echo "Start training on dataset ${dataset}......"
        python train_model_v2.py ${all_datasets[@]} --leave_one_out 0 -lr1 1e-5 -lr2 1e-5 -d 0.1 -s ${seed} -clipgrad True -step 3 --earlystopping_step 5 -dataset ${dataset} -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data}
    done
done

# # ZeroStance ablation on prompts, e.g., only using prompt 2
# config=../config/config-roberta_large.txt
# train_data=../data/
# dev_data=../data/
# test_data=../data/
# all_datasets=("vast" "ibm30k" "covid19" "semeval2016" "wtwt" "pstance")
# for seed in 4 5 7
# do
#     echo "Start training on seed ${seed}......"
#     for dataset in "chatgpt_carto_var_0.99_1type_2_seed0"
#     do
#         echo "Start training on dataset ${dataset}......"
#         python train_model_v2.py ${all_datasets[@]} --leave_one_out 0 -lr1 1e-5 -lr2 1e-5 -d 0.1 -s ${seed} -clipgrad True -step 3 --earlystopping_step 5 -dataset ${dataset} -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data}
#     done
# done

# # ZeroStance downscaling subset, e.g., only using 500 samples
# config=../config/config-roberta_large.txt
# train_data=../data/
# dev_data=../data/
# test_data=../data/
# all_datasets=("vast" "ibm30k" "covid19" "semeval2016" "wtwt" "pstance")
# for seed in 4 5 7
# do
#     echo "Start training on seed ${seed}......"
#     for dataset in "chatgpt_carto_var_0.99_seed0_subset_500"
#     do
#         echo "Start training on dataset ${dataset}......"
#         python train_model_v2.py ${all_datasets[@]} --leave_one_out 0 -lr1 1e-5 -lr2 1e-5 -d 0.1 -s ${seed} -clipgrad True -step 3 --earlystopping_step 5 -dataset ${dataset} -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data}
#     done
# done

# # ZeroStance as augment dataset, i.e., train on ZeroStance+5datasets and eval on the rest one
# config=../config/config-roberta_large.txt
# train_data=../data/
# dev_data=../data/
# test_data=../data/
# all_datasets=("covid19" "pstance" "semeval2016" "vast" "wtwt" "ibm30k" "chatgpt")
# for seed in 4 5 7
# do
#     echo "Start training on seed ${seed}......"
#     for dataset in "covid19" "pstance" "semeval2016" "vast" "wtwt" "ibm30k"
#     do
#         echo "Start training on dataset ${dataset}......"
#         python -u train_model_v2.py ${all_datasets[@]} --leave_one_out 1 -lr1 1e-5 -lr2 1e-5 -d 0.1 -s ${seed} -clipgrad True -step 3 --earlystopping_step 5 -dataset ${dataset} -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data}
#     done
# done

# # RoBERTa in out-of-domain setup, leave-one-out setting
# config=../config/config-roberta_large.txt
# train_data=../data/
# dev_data=../data/
# test_data=../data/
# all_datasets=("covid19" "pstance" "semeval2016" "vast" "wtwt" "ibm30k")
# for seed in 4 5 7
# do
#     echo "Start training on seed ${seed}......"
#     for dataset in "covid19" "pstance" "semeval2016" "vast" "wtwt" "ibm30k"
#     do
#         echo "Start training on dataset ${dataset}......"
#         python -u train_model_v2.py ${all_datasets[@]} --leave_one_out 1 -lr1 1e-5 -lr2 1e-5 -d 0.1 -s ${seed} -clipgrad True -step 3 --earlystopping_step 5 -dataset ${dataset} -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data}
#     done
# done

# # OpenStance with vast
# config=../config/config-roberta_large.txt
# train_data=../data/
# dev_data=../data/
# test_data=../data/
# all_datasets=("vast" "ibm30k" "covid19" "semeval2016" "wtwt" "pstance")
# for seed in 4 5 7
# do
#     echo "Start training on seed ${seed}......"
#     for dataset in "openstance_vast"
#     do
#         echo "Start training on dataset ${dataset}......"
#         python train_model_v2.py ${all_datasets[@]} --leave_one_out 0 -lr1 1e-5 -lr2 1e-5 -d 0.1 -s ${seed} -clipgrad True -step 3 --earlystopping_step 5 -dataset ${dataset} -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data}
#     done
# done
###################################################################################################################
###################################################################################################################
###################################################################################################################
