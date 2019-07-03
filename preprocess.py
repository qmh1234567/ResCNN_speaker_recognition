#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# __author__: Qmh
# __file_name__: preprocess.py
# __time__: 2019:06:27:16:23
import os
import librosa
import numpy as np
import pandas as pd
from progress.bar import Bar
import constants as c
import pandas as pd
from scipy.signal import lfilter,butter
import sigproc
import matplotlib.pyplot as plt
import logging
import math
import python_speech_features as psf
from python_speech_features import logfbank

class SilenceDetector(object):
    def __init__(self, threshold=20, bits_per_sample=16):
        self.cur_SPL = 0
        self.threshold = threshold
        self.bits_per_sample = bits_per_sample
        self.normal = pow(2.0, bits_per_sample - 1);
        self.logger = logging.getLogger('balloon_thrift')

    def is_silence(self, chunk):
        self.cur_SPL = self.soundPressureLevel(chunk)
        is_sil = self.cur_SPL < self.threshold
        # print('cur spl=%f' % self.cur_SPL)
        if is_sil:
            self.logger.debug('cur spl=%f' % self.cur_SPL)
        return is_sil

    def soundPressureLevel(self, chunk):
        value = math.pow(self.localEnergy(chunk), 0.5)
        value = value / len(chunk) + 1e-12
        value = 20.0 * math.log(value, 10)
        return value

    def localEnergy(self, chunk):
        power = 0.0
        for i in range(len(chunk)):
            sample = chunk[i] * self.normal
            power += sample*sample
        return power


# remove VAD
def VAD_audio(wav,sr,threshold = 15):
    sil_detector = SilenceDetector(threshold)
    new_wav = []
    if sr != 16000:
        wav = librosa.resample(wav, sr, 16000)
        sr = 16000
    for i in range(int(len(wav)/(sr*0.02))):
        start = int(i*sr*0.02)
        end = start + int(sr*0.02)
        is_silence = sil_detector.is_silence(wav[start:end])
        if not is_silence:
            new_wav.extend(wav[start:end])
    return new_wav


# comput mel_spectrogram
def compute_melgram(audio_path):
    src , sr = librosa.load(audio_path,c.SR)
    n_sample = src.shape[0]
    singal_len = int(c.DURA*c.SR)
    if n_sample < singal_len:
        src = np.hstack((src,np.zeros(singal_len-n_sample)))
    else:
        src = src[(n_sample-singal_len)//2:(n_sample+singal_len)//2]
    #  shape =(n_mels, (second*sr)/ hop_length ) freuency and time
    melgram = librosa.feature.melspectrogram(y=src,sr= c.SR,hop_length=c.HOP_LEN,n_fft=c.N_FFT,n_mels = c.N_MELS)
    ret = librosa.amplitude_to_db(melgram)
    ret = ret[:,:,np.newaxis]
    return ret

# mean and variance normalization 
# https://github.com/christianvazquez7/ivector/blob/master/MSRIT/rm_dc_n_dither.m
def normalization_frames(m,epsilon=1e-12):
    return np.array([(v-np.mean(v))/max(np.std(v),epsilon) for v in m ])

def remove_dc_and_dither(sin,sample_rate):
    if sample_rate == 16e3:
        alpha = 0.99
    elif sample_rate == 8e3:
        alpha = 0.999
    else:
        print("Sample rate must be 16kHZ or 8kHZ only")
        exit(1)
    sin = lfilter([1,-1],[-1,-alpha],sin)
    dither = np.random.random_sample(len(sin))+np.random.random_sample(len(sin))-1
    spow = np.std(dither)
    sout = sin + 1e-6*spow *dither
    return sout

def get_fft_spectrum(audio_path):
    signal,sr = librosa.load(audio_path,sr= c.SR)
    # padding zero
    n_sample = signal.shape[0]
    singal_len = int(c.DURA*c.SR)
    if n_sample < singal_len:
        signal = np.hstack((signal,np.zeros(singal_len-n_sample)))
    else:
        signal = signal[(n_sample-singal_len)//2:(n_sample+singal_len)//2]
    signal *= 2**15
    signal = remove_dc_and_dither(signal,c.SR)
    signal = sigproc.preemphasis(signal,coeff=c.PREEMPHASIS_ALPHA)
    frames = sigproc.framesig(signal,frame_len=c.FRAME_LEN*c.SR,frame_step=c.FRAME_STEP*c.SR,winfunc=np.hamming)
    fft = abs(np.fft.fft(frames,n= c.N_FFT))
    fft_norm = normalization_frames(fft.T)
    return fft_norm

def extract_feature(audio_path):
    signal,sr = librosa.load(audio_path,sr=c.SR)
    # remove vad
    signal = np.array(VAD_audio(signal.flatten(),sr,15))
    # padding zero
    n_sample = signal.shape[0]
    singal_len = int(c.DURA*c.SR)
    if n_sample < singal_len:
        signal = np.hstack((signal,np.zeros(singal_len-n_sample)))
    else:
        signal = signal[(n_sample-singal_len)//2:(n_sample+singal_len)//2]
    # extract fbank feature
    feat = psf.logfbank(signal,c.SR,nfilt=64)
    # normalize
    feat = np.array(normalization_frames(feat))
    feat1 = psf.delta(feat,1)
    feat2 = psf.delta(feat,2)
    feat = feat.T[:,:,np.newaxis]
    feat1 = feat1.T[:,:,np.newaxis]
    feat2 = feat2.T[:,:,np.newaxis]
    fbank = np.concatenate((feat,feat1,feat2),axis=2)
    return fbank

    
#  save audio path to txt
def Write_path_to_csv(dataset,speakers,typeName):
    if not os.path.exists(c.DATASET_DIR):
        os.mkdir(c.DATASET_DIR)
    audio_paths = []
    speaker_id = []
    for speaker in speakers:
        video_files = os.listdir(f'{dataset}/wav/{speaker}')
        for video in video_files:
            wavfiles = os.listdir(f'{dataset}/wav/{speaker}/{video}')
            for wav in wavfiles:
                if wav.endswith('.wav'):
                    wavpath = f'{dataset}/wav/{speaker}/{video}/{wav}'
                    audio_paths.append(wavpath)
                    speaker_id.append(speaker)
    dict_all ={
        'FilePath':audio_paths,
        'SpeakerID':speaker_id
    }
    data = pd.DataFrame(dict_all)
    data.to_csv(f'{c.DATASET_DIR}/{typeName}.csv',index=0)
    print(f"{c.DATASET_DIR}/{typeName}.csv succeed")
    return audio_paths


# preprocess
def preprocess_to_npy(dataset,num_class,typeName):
    if typeName.startswith('train'):
        speakers = os.listdir(f'{dataset}/wav')[:num_class]
    else:
        speakers = os.listdir(f'{dataset}/wav')
    # create npy folder
    if not os.path.exists(f'{dataset}/npy'):
        os.mkdir(f'{dataset}/npy')
    #  read audio paths
    audio_paths = Write_path_to_csv(dataset,speakers,typeName)
    # extract features
    bar = Bar("Processing",max=len(audio_paths),fill="#",suffix='%(percent)d%%')
    for audio in audio_paths:
        bar.next()
        audio_info = os.path.splitext(audio)[0].split('/')[-3:]
        audio_name = '_'.join(audio_info)
        try:
            speaker_dir = f'{dataset}/npy/{audio_info[0]}'
            if  os.path.exists(speaker_dir+f'/{audio_name}.npy'):
                continue
            # melgram = get_fft_spectrum(audio)
            melgram = extract_feature(audio) 
            if not os.path.exists(speaker_dir):
                os.mkdir(speaker_dir)
            np.save(speaker_dir+f'/{audio_name}.npy',melgram)
        except Exception as e:
            print(e)
    bar.finish()


if __name__ == "__main__":
    # voxceleb
#    preprocess_to_npy(c.TRAIN_DEV_SET,c.CLASS,'train_vox')
#    preprocess_to_npy(c.TEST_SET,c.CLASS,'test_vox')
    #  library speech
    preprocess_to_npy(c.TRAIN_DEV_SET_LB,c.CLASS,'train_lb')
    preprocess_to_npy(c.TEST_SET_LB,c.CLASS,'test_lb')
    # npyfile = '/home/dsp/Documents/wav_data/SpeakerRecognition_dataset/Library_speech_100/train-clean-100/LibriSpeech/train-clean-100/npy/19/19_198_19-198-0000.npy'
    # print(np.load(npyfile).shape)


