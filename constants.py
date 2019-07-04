#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# __author__: Qmh
# __file_name__: constants.py
# __time__: 2019:06:27:16:18

# MEL-SPECTROGRAM
SR = 16000
DURA = 3  # 3s
N_FFT = 512
N_MELS = 128  
HOP_LEN = 161
# SIGNAL PROCESSING
PREEMPHASIS_ALPHA = 0.97
FRAME_LEN = 0.025  
FRAME_STEP = 0.01  

# DATASET
TRAIN_DEV_SET = '/home/dsp/Documents/wav_data/SpeakerRecognition_dataset/Voxceleb/vox1_dev_wav/'
TRAIN_DEV_SET_LB = '/home/dsp/Documents/wav_data/SpeakerRecognition_dataset/Library_speech_100/train-clean-100/LibriSpeech/train-clean-100'

TEST_SET = '/home/dsp/Documents/wav_data/SpeakerRecognition_dataset/Voxceleb/vox_test_wav'
TEST_SET_LB = '/home/dsp/Documents/wav_data/SpeakerRecognition_dataset/Library_speech_100/test-clean/LibriSpeech/test-clean'

TARGET = 'SV'
ENROLL_NUMBER = 20

CLASS = 251
DATASET_DIR = './dataset'

# MODEL
WEIGHT_DECAY = 0.00001

# TRAIN
BATCH_SIZE = 16
# INPUT_SHPE = (128,299,1)
# frequency time channel
INPUT_SHPE = (64,299,3)
MODEL_DIR ='./checkpoint'
LEARN_RATE = 0.0001

#TEST
ANNONATION_FILE = './dataset/annonation.csv'
THRESHOLD = 0.66