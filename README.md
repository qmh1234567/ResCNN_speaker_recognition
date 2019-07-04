# ResCNN_speaker_recognition
运行准备：
1. 下载library_speech数据集，地址：http://www.openslr.org/resources/12/。共三个文件夹：	train-clean-100.tar.gz, test-clean.tar.gz,	dev-clean.tar.gz。
2. 将convert_flac_2_wav_sox.sh复制到数据集所在目录，该文件是将.flac转化为.wav. 

  reference to:https://github.com/Walleclipse/Deep_Speaker-speaker_recognition_system
  
3. 在train-clean和test-clean目录下新建wav文件夹，将所有音频文件都剪切进去。

运行说明:
1. 修改constants.py 中的数据集路径:TRAIN_DEV_SET_LB和TEST_SET_LB,改成您下载的library speech的相应目录.
2. 运行preprocess.py,特征文件生成在下载的library speech的对应目录下的npy文件夹.
3. 运行python run.py train_lb命令进行训练模型,模型有两个:rescnn和deepspeaker,目前两个的SI效果都不太好.
4. 运行python run.py test_lb命令进行测试,设置constants.py中的TARGET切换SI或SV.
