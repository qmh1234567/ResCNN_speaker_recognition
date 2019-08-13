#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# __author__: Qmh
# __file_name__: run.py
# __time__: 2019:06:27:20:53

import pandas as pd
import constants as c
import os
import tensorflow as tf
from collections import Counter
import numpy as np
from progress.bar import Bar
import models
from keras.layers import Dense
from keras import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import sys
import keras.backend.tensorflow_backend as KTF
from keras.layers import Flatten
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedKFold
from keras import optimizers
import glob
import pickle

# OPTIONAL: control usage of GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.Session(config=config)


def shuffle_data(paths, labels):
    length = len(labels)
    shuffle_index = np.arange(0, length)
    shuffle_index = np.random.choice(shuffle_index, size=length, replace=False)
    paths = np.array(paths)[shuffle_index]
    labels = np.array(labels)[shuffle_index]
    return paths, labels


def split_perspeaker_audios(audio_paths, audio_labels, split_ratio=0.1):
    val_paths, val_labels, train_paths, train_labels = [], [], [], []
    dict_count = Counter(audio_labels)
    for speaker in set(audio_labels):
        start = audio_labels.index(speaker)
        end = start + dict_count[speaker]
        # shuffle
        np.random.shuffle(audio_paths[start:end])
        for index in range(start, end):
            if index < start+dict_count[speaker]*split_ratio:
                val_paths.append(audio_paths[index])
                val_labels.append(speaker)
            else:
                train_paths.append(audio_paths[index])
                train_labels.append(speaker)
    return val_paths, val_labels, train_paths, train_labels


def CreateDataset(typeName, split_ratio=0.2, target='SV'):
    train_paths, val_paths = [], []
    train_labels, val_labels = [], []
    seed = 42
    np.random.seed(seed)
    # shuffle
    if typeName.startswith('train'):
        speaker_pickle_files_list = [
            pickle for pickle in glob.iglob(c.PICKLE_TRAIN_DIR + "/*.pickle")]
        audio_labels = [os.path.basename(pickle).split(
            "_")[0] for pickle in speaker_pickle_files_list]
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            speaker_pickle_files_list, audio_labels, stratify=audio_labels, test_size=split_ratio, random_state=42)
    else:
        audio_paths = [pickle for pickle in glob.iglob(
            c.PICKLE_TEST_DIR+"/*.pickle")]
        audio_paths.sort()
        audio_labels = [os.path.basename(pickle).split(
            "_")[0] for pickle in audio_paths]
        # 7：3 per speaker
        if target == 'SI':
            val_paths, val_labels, train_paths, train_labels = split_perspeaker_audios(
                audio_paths, audio_labels, split_ratio)
        else:
            dict_count = Counter(audio_labels)
            cut_index = 0
            # remove repeat and rank in order
            new_audio_labels = list(set(audio_labels))
            new_audio_labels.sort(key=audio_labels.index)
            for index, speaker in enumerate(new_audio_labels):
                if index == c.ENROLL_NUMBER:
                    break
                else:
                    cut_index += dict_count[speaker]
            val_paths, val_labels, train_paths, train_labels = split_perspeaker_audios(
                audio_paths[:cut_index], audio_labels[:cut_index], split_ratio)
            ismember = np.ones(len(train_labels)).tolist()
            # non-speaker
            train_paths.extend(audio_paths[cut_index:])
            train_labels.extend(audio_labels[cut_index:])
            # is member
            ismember.extend(np.zeros(len(audio_labels[cut_index:])).tolist())
            # shuffle
            length = len(train_paths)
            shuffle_index = np.arange(0, length)
            shuffle_index = np.random.choice(
                shuffle_index, size=length, replace=False)
            train_paths = np.array(train_paths)[shuffle_index]
            train_labels = np.array(train_labels)[shuffle_index]
            ismember = np.array(ismember)[shuffle_index]
            # create annonation.csv
            data_dict = {
                'FilePath': train_paths,
                'SpeakerID': train_labels,
                'Ismember': ismember,
            }
            data = pd.DataFrame(data_dict)
            data.to_csv(c.ANNONATION_FILE, index=0)
            print(f"wirte to {c.ANNONATION_FILE} succeed")

    train_dataset = (train_paths, train_labels)
    val_dataset = (val_paths, val_labels)
    print("len(train_paths)=", len(train_paths))
    print("len(val_paths)=", len(val_paths))
    print("len(audio_paths)=", len(audio_labels))
    print("len(set(train_labels))=", len(set(train_labels)))
    return train_dataset, val_dataset


def Map_label_to_dict(labels):
    labels_to_id = {}
    i = 0
    for label in np.unique(labels):
        labels_to_id[label] = i
        i += 1
    return labels_to_id

def load_validation_data(dataset, labels_to_id, num_class):
    (path, labels) = dataset
    path, labels = shuffle_data(path, labels)
    X, Y = [], []
    bar = Bar('loading data', max=len(labels),
              fill='#', suffix='%(percent)d%%')
    for index, pk in enumerate(path):
        bar.next()
        try:
            with open(pk, "rb") as f:
                load_dict = pickle.load(f)
                x = load_dict["LogMel_Features"]
                x = x[:, :, np.newaxis]
                X.append(x)
                Y.append(labels_to_id[labels[index]])
        except Exception as e:
            print(e)
    X = np.array(X)
    Y = np.eye(num_class)[Y]
    bar.finish()
    return (X, Y)


def load_all_data(dataset, typeName):
    (path, labels) = dataset
    X, Y = [], []
    bar = Bar('loading data', max=len(path), fill='#', suffix='%(percent)d%%')
    for index, audio in enumerate(path):
        bar.next()
        try:
            with open(audio, "rb") as f:
                load_dict = pickle.load(f)
                x = load_dict["LogMel_Features"]
                x = x[:, :, np.newaxis]
                X.append(x)
                Y.append(labels[index])
        except Exception as e:
            print(e)
    bar.finish()
    return (np.array(X), np.array(Y))


def load_each_batch(dataset, labels_to_id, batch_start, batch_end, num_class):
    (paths, labels) = dataset
    X, Y = [], []
    for i in range(batch_start, batch_end):
        try:
            with open(paths[i], "rb") as f:
                load_dict = pickle.load(f)
                x = load_dict["LogMel_Features"]
                x = x[:, :, np.newaxis]
                X.append(x)
                Y.append(labels_to_id[labels[i]])
        except Exception as e:
            print(e)
    X = np.array(X)
    Y = np.eye(num_class)[Y]
    return X, Y


def Batch_generator(dataset, labels_to_id, batch_size, num_class):
    (paths, labels) = dataset
    length = len(labels)
    while True:
        # shuffle
        paths, labels = shuffle_data(paths, labels)
        shuffle_dataset = (paths, labels)
        batch_start = 0
        batch_end = batch_size
        while batch_end < length:
            X, Y = load_each_batch(
                shuffle_dataset, labels_to_id, batch_start, batch_end, num_class)
            yield (X, Y)
            batch_start += batch_size
            batch_end += batch_size


def caculate_distance(enroll_dataset, enroll_pre, test_pre):
    print("enroll_pre.shape=", enroll_pre.shape)
    dict_count = Counter(enroll_dataset[1])
    print(dict_count)
    # each person get a enroll_pre
    speakers_pre = []
    # remove repeat
    enroll_speakers = list(set(enroll_dataset[1]))
    enroll_speakers.sort(key=enroll_dataset[1].index)
    for speaker in enroll_speakers:
        start = enroll_dataset[1].index(speaker)
        speaker_pre = enroll_pre[start:dict_count[speaker]+start]
        speakers_pre.append(np.mean(speaker_pre, axis=0))

    enroll_pre = np.array(speakers_pre)
    print("new_enroll_pre.shape=", enroll_pre.shape)
    # caculate distance
    distances = []
    print("test_pre.shape=", test_pre.shape)
    for i in range(enroll_pre.shape[0]):
        temp = []
        for j in range(test_pre.shape[0]):
            x = enroll_pre[i]
            y = test_pre[j]
            s = np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))
            temp.append(s)
        distances.append(temp)
    distances = np.array(distances)
    print("distances.shape=", distances.shape)
    return distances


def speaker_identification(enroll_dataset, distances, enroll_y):
    #  remove repeat
    new_enroll_y = list(set(enroll_y))
    new_enroll_y.sort(key=list(enroll_y).index)
    #  return the index of max distance of each sentence
    socre_index = distances.argmax(axis=0)
    y_pre = []
    for i in socre_index:
        y_pre.append(new_enroll_y[i])
    return y_pre


def compute_result(y_pre, y_true):
    result = []
    for index, x in enumerate(y_pre):
        result.append(1 if x == y_true[index] else 0)
    return result


def evaluate_metrics(y_true, y_pre):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pre, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='green', label='ROC')
    plt.plot(np.arange(1, 0, -0.01), np.arange(0, 1, 0.01))
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title(f'ROC curve, AUC score={auc}')
    plt.show()

    threshold_index = np.argmin(abs(1-tpr - fpr))
    threshold = thresholds[threshold_index]
    eer = ((1-tpr)[threshold_index]+fpr[threshold_index])/2
    print(eer)
    auc_score = metrics.roc_auc_score(y_true, y_pre, average='macro')

    y_pro = [1 if x > threshold else 0 for x in y_pre]
    acc = metrics.accuracy_score(y_true, y_pro)
    prauc = metrics.average_precision_score(y_true, y_pro, average='macro')
    return y_pro, eer, prauc, acc, auc_score


def speaker_verification(distances, ismember_true):
    score_index = distances.argmax(axis=0)
    distance_max = distances.max(axis=0)
    distance_max = (distance_max + 1) / 2
    y_pro, eer, prauc, acc, auc_score = evaluate_metrics(
        ismember_true, distance_max)
    print(f'eer={eer}\t prauc={prauc} \t acc={acc}\t auc_score={auc_score}\t')
    return y_pro

def main(typeName):
    # model = models.Deep_speaker_model(c.INPUT_SHPE)
    model = models.SE_ResNet(c.INPUT_SHPE)
    # model = models.Baseline_GRU(c.INPUT_SHPE)

    if typeName.startswith('train'):
        if not os.path.exists(c.MODEL_DIR):
            os.mkdir(c.MODEL_DIR)
        train_dataset, val_dataset = CreateDataset(typeName, split_ratio=0.1)
        nclass = len(set(train_dataset[1]))
        print("nclass = ",nclass)
     
        labels_to_id = Map_label_to_dict(labels=train_dataset[1])
        # add softmax layer
        x = model.output
        x = Dense(nclass, activation='softmax', name=f'softmax')(x)
        model = Model(model.input, x)
        model.summary()
 
        exit()
        # train model
        sgd = optimizers.SGD(lr=c.LEARN_RATE,momentum=0.9) #TIMIT libri-seresnet
        model.compile(loss='categorical_crossentropy', optimizer=sgd,
                      metrics=['accuracy'])
        model.fit_generator(Batch_generator(train_dataset, labels_to_id, c.BATCH_SIZE, nclass),
                            steps_per_epoch=len(train_dataset[0])//c.BATCH_SIZE, epochs=50,
                            validation_data=load_validation_data(
                                val_dataset, labels_to_id, nclass),
                            validation_steps=len(val_dataset[0])//c.BATCH_SIZE,
                            callbacks=[
            ModelCheckpoint(f'{c.MODEL_DIR}/best.h5',
                            monitor='val_loss', save_best_only=True, mode='min'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=10, mode='min'),
            EarlyStopping(monitor='val_loss', patience=10),
        ])
    else:
        model.summary()
        test_dataset, enroll_dataset = CreateDataset(
            typeName, split_ratio=0.5, target=c.TARGET)
        labels_to_id = Map_label_to_dict(labels=enroll_dataset[1])
        # load weights
        model.load_weights(f'{c.MODEL_DIR}/best.h5', by_name='True')
        # load all data
        print("loading data...")
        (enroll_x, enroll_y) = load_all_data(enroll_dataset, 'enroll')
        (test_x, test_y) = load_all_data(test_dataset, 'test')
        enroll_pre = np.squeeze(model.predict(enroll_x))
        test_pre = np.squeeze(model.predict(test_x))
        distances = caculate_distance(enroll_dataset, enroll_pre, test_pre)
        if c.TARGET == 'SI':
            # speaker identification
            test_y_pre = speaker_identification(
                enroll_dataset, distances, enroll_y)
            # compute result
            result = compute_result(test_y_pre, test_y)
            score = sum(result)/len(result)
            print(f"score={score}")
        elif c.TARGET == 'SV':
            df = pd.read_csv(c.ANNONATION_FILE)
            ismember_true = list(map(int, df['Ismember']))
            # np.save('./npys/perfect_noELU.npy',distances)
            ismember_pre = speaker_verification(distances, ismember_true)
        else:
            print("you should set the c.TARGET to SI and SV")
            exit(-1)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ['train', 'test']:
        print('Usage: python run.py [run_type]\n',
              '[run_type]: train | test')   
        exit()
    mode = sys.argv[1]
    main(mode)
