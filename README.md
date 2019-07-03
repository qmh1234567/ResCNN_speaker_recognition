# ResCNN_speaker_recognition
运行说明:
1.修改constants.py 中的数据集路径:TRAIN_DEV_SET_LB和TEST_SET_LB,改成您下载的library speech的相应目录.
2.运行preprocess.py,特征文件生成在下载的library speech的对应目录下的npy文件夹.
3.运行python run.py train_lb命令进行训练模型,模型有两个:rescnn和deepspeaker,目前两个的SI效果都不太好.
4.运行python run.py test_lb命令进行测试,设置constants.py中的TARGET切换SI或SV.
