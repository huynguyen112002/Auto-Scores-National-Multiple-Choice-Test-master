import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import numpy as np


class CNN_Model(object):
    def __init__(self, weight_path=None):
        self.weight_path = weight_path
        self.model = None

    def build_model(self, rt=False):  # xay dung mo hinh cnn
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='softmax'))

        if self.weight_path is not None:
            self.model.load_weights(self.weight_path)
        
        if rt:
            return self.model

    @staticmethod
    def load_data():  # nap hinh anh tu datasets
        dataset_dir = './datasets/'
        images = []
        labels = []

        for img_path in Path(dataset_dir + 'unchoice/').glob("*.png"):  # nap hinh anh tu thu muc unchoice
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
            img = img.reshape((28, 28, 1))
            label = to_categorical(0, num_classes=2)
            images.append(img / 255.0)
            labels.append(label)

        for img_path in Path(dataset_dir + 'choice/').glob("*.png"):  # nap hinh anh tu thu muc choice
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
            img = img.reshape((28, 28, 1))
            label = to_categorical(1, num_classes=2)
            images.append(img / 255.0)
            labels.append(label)

        datasets = list(zip(images, labels))
        np.random.shuffle(datasets)
        images, labels = zip(*datasets)
        images = np.array(images)
        labels = np.array(labels)
        
        return images, labels


    def train(self):  # huan luyen mo hinh cnn
        images, labels = self.load_data()

        # build model
        self.build_model(rt=False)

        # compile bien dich mo hinh
        self.model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(1e-3),
                           metrics=['acc'])

        # reduce learning rate Giảm tốc độ học nếu độ chính xác không cải thiện.
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5,
                                      verbose=1, )

        # Model Checkpoint Lưu lại mô hình có hiệu suất tốt nhất dựa trên độ chính xác của tập validation.
        cpt_save = ModelCheckpoint('./weight.h5', save_best_only=True, monitor='val_acc',
                                   mode='max')

        self.model.fit(images, labels, callbacks=[cpt_save, reduce_lr], verbose=1, epochs=10,
                       validation_split=0.15, batch_size=32,
                       shuffle=True)  # phan chia du lieu 85% cho train, 15% cho test

    print("Đã train thành công, kết quả được lưu tại weight.h5")

