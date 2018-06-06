# coding=utf8
from utils.drn import Residual_DenseNet
from keras.optimizers import SGD
import numpy as np
import cv2
import os
from sklearn.svm import LinearSVC
import random
from sklearn.preprocessing import normalize
from tqdm import tqdm


def svc(traindata, trainlabel, testdata, testlabel):
    print("Start training SVM...")
    svcClf = LinearSVC(C=10.0)
    svcClf.fit(traindata, trainlabel)

    pred_testlabel = svcClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i] == pred_testlabel[i]]) / float(num)
    print("svm Accuracy:", accuracy)


def main():
    root_path = '/home/deep/datasets/hmdb/'
    feat = []
    labels = []
    clip_length = 16

    # init model
    model = Residual_DenseNet(nb_classes=101, input_shape=(112, 112, 8, 3), extract_feat=True)
    lr = 0.005
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    model.load_weights('./results/drn-16/weights_drn2.h5', by_name=True)

    f = open('./ucfTrainTestlist/hmdb.txt', 'r')
    samples = f.readlines()
    pbar = tqdm(total=len(samples))
    for sample in samples:
        pbar.update(1)
        sample = sample.strip()
        label = int(sample.split(' ')[-1])
        sample_path = root_path + sample.split(' ')[0]
        imgs = os.listdir(sample_path)
        L = len(imgs)
        imgs.sort(key=str.lower)
        step = L // clip_length
        tmp_feat = np.zeros(512, dtype=np.float32)
        for s in range(step):
            clip = []
            for l in range(clip_length):
                if l % 2 == 0:
                    index = s * clip_length + l
                    img = cv2.imread(sample_path + '/' + imgs[index])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (171, 128))
                    clip.append(img)
            clip = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(clip, axis=0)
            inputs[..., 0] -= 99.9
            inputs[..., 1] -= 92.1
            inputs[..., 2] -= 82.6
            inputs[..., 0] /= 65.8
            inputs[..., 1] /= 62.3
            inputs[..., 2] /= 60.3
            inputs = inputs[:, :, 8:120, 30:142, :]
            inputs = np.transpose(inputs, (0, 2, 3, 1, 4))
            pred = model.predict(inputs)
            tmp_feat += pred[0]
        tmp_feat = np.array(tmp_feat).astype(np.float32)
        tmp_feat /= step

        feat.append(tmp_feat)
        labels.append(label)
    f.close()
    pbar.close()
    index = [i for i in range(len(feat))]
    random.shuffle(index)
    feat = np.array(feat)
    labels = np.array(labels)
    feat = feat[index]
    feat = normalize(feat, norm='l2')
    labels = labels[index]
    svc(feat[0:len(feat) * 0.8], labels[0:len(feat) * 0.8], feat[len(feat) * 0.8:], labels[len(feat) * 0.8:])


if __name__ == '__main__':
    main()
