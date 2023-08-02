export CUDA_VISIBLE_DEVICES=1
python utils/make_sound_dataset.py --processed_method intensity --save_dir dataset
python utils/make_csv_files.py --dataset_name pose_regression_timeseries --subject_name subject_1 --sound_csv_path ./dataset/sound_intensity.csv --preprocess intensity
python utils/make_csv_files.py --dataset_name pose_regression_timeseries --subject_name subject_2 --sound_csv_path ./dataset/sound_intensity.csv --preprocess intensity
python utils/make_csv_files.py --dataset_name pose_regression_timeseries --subject_name subject_3 --sound_csv_path ./dataset/sound_intensity.csv --preprocess intensity
python utils/make_csv_files.py --dataset_name pose_regression_timeseries --subject_name subject_4 --sound_csv_path ./dataset/sound_intensity.csv --preprocess intensity
python utils/make_csv_files.py --dataset_name pose_regression_timeseries --subject_name subject_5 --sound_csv_path ./dataset/sound_intensity.csv --preprocess intensity
python utils/make_csv_files.py --dataset_name pose_regression_timeseries --subject_name subject_6 --sound_csv_path ./dataset/sound_intensity.csv --preprocess intensity
python utils/make_csv_files.py --dataset_name pose_regression_timeseries --subject_name subject_7 --sound_csv_path ./dataset/sound_intensity.csv --preprocess intensity
python utils/make_csv_files.py --dataset_name pose_regression_timeseries --subject_name subject_8 --sound_csv_path ./dataset/sound_intensity.csv --preprocess intensity

python utils/make_configs.py --batch_size 256 --learning_rate 0.003 --max_epoch 100 --sound_length 2400 --input_feature logmel logmel --dataset_name pose_regression_timeseries_subject_1 pose_regression_timeseries_subject_2 pose_regression_timeseries_subject_3 pose_regression_timeseries_subject_4 pose_regression_timeseries_subject_5 pose_regression_timeseries_subject_6 pose_regression_timeseries_subject_7 pose_regression_timeseries_subject_8 --model speech2pose speech2pose --smooth_loss True True 

files="./result/*timeseries*"
for filepath in $files; do
    if [ -d $filepath ] ; then
        flag="${filepath}/final_model.prm"
        if [ -e $flag ] ; then
            continue
        fi
        python train.py "${filepath}/config.yaml" --use_wandb
        python evaluate.py "${filepath}/config.yaml" validation
        python evaluate.py "${filepath}/config.yaml" test
    fi
done
