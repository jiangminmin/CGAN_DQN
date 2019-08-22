# CGAN_DQN
#pix2pix数据集可自己采集png图片制作\
#pix2pix的代码运行命令\
python pix2pix.py \
  --mode train \
  --output_dir facades_train \
  --max_epochs 200 \
  --input_dir facades/train \
  --which_direction BtoA
#pix2pix调用tensorboard命令\
tensorboard --logdir=facades_train\

#dgn-cgan代码运行命令\
python dqn-cgn.py\
#dgn-cgan调用tensorboard命令\
tensorboard --logdir=summary/

#my_flask_api文件夹是flask部署CGAN模型的代码\
#终端执行命令，可获取模型输出数据：curl --form "file=@upload.jpg" http://10.20.51.223:5000/api \
#代码访问模型：
url = "http://10.20.51.223:5000/api" \
##data = requests.get(url).json \
files = {'file': open('test.jpg', 'rb')} \
r = requests.post(url, files=files) \
result = r.text \
#1.文件目录用绝对路径\
#2.不要用pycharm运行app.py代码，直接在终端运行
