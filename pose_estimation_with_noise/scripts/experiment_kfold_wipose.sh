export CUDA_VISIBLE_DEVICES=0
python utils/make_sound_dataset.py --processed_method raw --save_dir dataset_raw
python utils/make_csv_files.py --dataset_name pose_regression_kfold --subject_name subject_1 --sound_csv_path ./dataset_raw/sound_raw.csv --preprocess raw
python utils/make_csv_files.py --dataset_name pose_regression_kfold --subject_name subject_2 --sound_csv_path ./dataset_raw/sound_raw.csv --preprocess raw
python utils/make_csv_files.py --dataset_name pose_regression_kfold --subject_name subject_3 --sound_csv_path ./dataset_raw/sound_raw.csv --preprocess raw
python utils/make_csv_files.py --dataset_name pose_regression_kfold --subject_name subject_4 --sound_csv_path ./dataset_raw/sound_raw.csv --preprocess raw
python utils/make_csv_files.py --dataset_name pose_regression_kfold --subject_name subject_5 --sound_csv_path ./dataset_raw/sound_raw.csv --preprocess raw
python utils/make_csv_files.py --dataset_name pose_regression_kfold --subject_name subject_6 --sound_csv_path ./dataset_raw/sound_raw.csv --preprocess raw
python utils/make_csv_files.py --dataset_name pose_regression_kfold --subject_name subject_7 --sound_csv_path ./dataset_raw/sound_raw.csv --preprocess raw
python utils/make_csv_files.py --dataset_name pose_regression_kfold --subject_name subject_8 --sound_csv_path ./dataset_raw/sound_raw.csv --preprocess raw
python utils/make_configs.py --batch_size 128 --learning_rate 0.001 --max_epoch 30 --sound_length 2400 --input_feature raw raw --dataset_name pose_regression_kfold_subject_1 pose_regression_kfold_subject_2 --model wipose_lstm wipose_lstm --smooth_loss True True
python utils/make_configs.py --batch_size 128 --learning_rate 0.001 --max_epoch 30 --sound_length 2400 --input_feature raw raw --dataset_name pose_regression_kfold_subject_3 pose_regression_kfold_subject_4 --model wipose_lstm wipose_lstm --smooth_loss True True
python utils/make_configs.py --batch_size 128 --learning_rate 0.001 --max_epoch 30 --sound_length 2400 --input_feature raw raw --dataset_name pose_regression_kfold_subject_5 pose_regression_kfold_subject_6 --model wipose_lstm wipose_lstm --smooth_loss True True
python utils/make_configs.py --batch_size 128 --learning_rate 0.001 --max_epoch 30 --sound_length 2400 --input_feature raw raw --dataset_name pose_regression_kfold_subject_7 pose_regression_kfold_subject_8 --model wipose_lstm wipose_lstm --smooth_loss True True
files="./result/*wipose_lstm*"
for filepath in $files; do
    if [ -d $filepath ] ; then
        flag="${filepath}/final_model.prm"
        if [ -e $flag ] ; then
            continue
        fi
        echo $filepath
        python train.py "${filepath}/config.yaml" --use_wandb
        python evaluate.py "${filepath}/config.yaml" validation
        python evaluate.py "${filepath}/config.yaml" test
    fi
done
