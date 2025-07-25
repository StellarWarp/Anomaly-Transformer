python main.py --anormly_ratio 1 --num_epochs 16 --batch_size 128  --mode train --dataset PSM  --data_path dataset/GAME_DATA --input_c 36    --output_c 36
python main.py --anormly_ratio 1 --num_epochs 10 --batch_size 128  --mode test  --dataset PSM  --data_path dataset/GAME_DATA  --input_c 36    --output_c 36  --pretrained_model 20

python main.py --num_epochs 10 --batch_size 128 --mode predict --dataset PSM --data_path dataset/GAME_DATA/test.csv  --input_c 36  --output_c 36  --pretrained_model 20 --output_path predictions/GAME_DATA.csv --threshold 0.23755799 --visualize True
