# coding=utf8
from utils.drn import Residual_DenseNet
from keras.optimizers import SGD
import numpy as np
import cv2


def main():
    with open('./ucfTrainTestlist/classInd.txt', 'r') as f:
        class_names = f.readlines()
        f.close()

    # init model
    model = Residual_DenseNet(nb_classes=101, input_shape=(112, 112, 8, 3))
    lr = 0.005
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    model.load_weights('./results/drn-16/weights_drn2.h5', by_name=True)

    # read video
    video = './videos/v_Punch_g03_c01.avi'
    cap = cv2.VideoCapture(video)

    clip1 = []
    clip2 = []
    clip_count = 0
    while True:
        ret, frame = cap.read()
        if ret:
            tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if clip_count % 2 == 0:
                clip1.append(cv2.resize(tmp, (171, 128)))
                if len(clip1) == 8:
                    inputs = np.array(clip1).astype(np.float32)
                    inputs = np.expand_dims(inputs, axis=0)
                    inputs[..., 0] -= 99.9
                    inputs[..., 1] -= 92.1
                    inputs[..., 2] -= 82.6
                    inputs[..., 0] /= 65.8
                    inputs[..., 1] /= 62.3
                    inputs[..., 2] /= 60.3
                    inputs = inputs[:, :, 8:120, 30:142, :]
                    inputs = np.transpose(inputs, (0, 2, 3, 1, 4))
                    pred = model.predict(inputs)
                    label = np.argmax(pred[0])
                    cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 1)
                    cv2.putText(frame, "prob: %.4f" % pred[0][label], (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 1)
                    clip1.pop(0)
            else:
                clip2.append(cv2.resize(tmp, (171, 128)))
                if len(clip2) == 8:
                    inputs = np.array(clip2).astype(np.float32)
                    inputs = np.expand_dims(inputs, axis=0)
                    inputs[..., 0] -= 99.9
                    inputs[..., 1] -= 92.1
                    inputs[..., 2] -= 82.6
                    inputs[..., 0] /= 65.8
                    inputs[..., 1] /= 62.3
                    inputs[..., 2] /= 60.3
                    inputs = inputs[:, :, 8:120, 30:142, :]
                    inputs = np.transpose(inputs, (0, 2, 3, 1, 4))
                    pred = model.predict(inputs)
                    label = np.argmax(pred[0])
                    cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 1)
                    cv2.putText(frame, "prob: %.4f" % pred[0][label], (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 1)
                    clip2.pop(0)
            clip_count += 1
            cv2.imshow('result', frame)
            cv2.waitKey(10)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
