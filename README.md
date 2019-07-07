# CGAN_DQN
#pix2pix数据集可自己采集png图片制作
#pix2pix的代码运行命令
python pix2pix.py \
  --mode train \
  --output_dir facades_train \
  --max_epochs 200 \
  --input_dir facades/train \
  --which_direction BtoA
#pix2pix调用tensorboard命令
tensorboard --logdir=facades_train

#dgn-cgan代码运行命令
python dqn-cgn.py
#dgn-cgan调用tensorboard命令
tensorboard --logdir=summary/
