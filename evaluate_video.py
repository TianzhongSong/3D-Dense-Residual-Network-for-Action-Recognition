from utils.drn import Residual_DenseNet
import numpy as np
import cv2
import os
from tqdm import tqdm

if __name__ == '__main__':
    root_path = '/home/deep/datasets/ucfimgs/'
    test_file = './ucfTrainTestlist/test_file.txt'

    top1_count = 0

    f = open(test_file, 'r')
    files = f.readlines()
    total_count = len(files)
    f.close()

    model = Residual_DenseNet(nb_classes=101, input_shape=(112, 112, 8, 3))
    model.load_weights('./results/drn-16/weights_drn2.h5')
    pbdr = tqdm(total=total_count)
    for file in files:
        pbdr.update(1)
        label = int(file.split(' ')[-1].strip())
        video_name = file.split(' ')[0]
        imgs = os.listdir(root_path + video_name)
        imgs.sort(key=str.lower)
        clip = []
        clip_count = 0
        acc = np.zeros(101, dtype=np.float32)
        for j in range(len(imgs)):
            if len(clip) == 8:
                clip = np.array(clip).astype(np.float32)
                clip = np.expand_dims(clip, axis=0)
                clip[..., 0] -= 99.9
                clip[..., 1] -= 92.1
                clip[..., 2] -= 82.6
                clip[..., 0] /= 65.8
                clip[..., 1] /= 62.3
                clip[..., 2] /= 60.3
                clip = np.transpose(clip, (0, 2, 3, 1, 4))
                pred = model.predict(clip)
                acc += pred[0]
                clip_count += 1
                clip = []
            else:
                if j % 2 == 0:
                    im = cv2.imread(root_path + video_name + '/' + imgs[j])
                    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                    im = cv2.resize(im, (171, 128))
                    clip.append(im[8:120, 30:142, :])
        acc /= clip_count
        pred = np.argmax(acc)
        if pred == label:
            top1_count += 1
    pbdr.close()
    print(top1_count)
    print(top1_count/total_count)
