# DRN-3D

3D dense residual network for action recognition

## No paper, just an experiment! 

Limited by hardware(I only have one GTX1080 Ti) and network(CN), I did not do further experiments with Large datastes, e.g [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/), [sports-8M](https://research.google.com/youtube8m/explore.html). 

## 3D dense residual network

Inspired by [Residual Dense Network for Image Super-Resolution](https://github.com/yulunzhang/RDN)

![3D dense residual block](https://github.com/TianzhongSong/DRN-3D/blob/master/imgs/3D-RDB.jpg)

                                    fig1 3D dense residual block

![3D dense redidual network](https://github.com/TianzhongSong/DRN-3D/blob/master/imgs/framework2.jpg)

                                    fig2 3D dense residual network

## Requirements

    opencv3.2
    keras2.0.8
    tensorflow1.3

## Prepare data

setp1 -- download [UCF-101](http://crcv.ucf.edu/data/UCF101.php) dataset

step2 -- converting videos to images for [UCF-101](http://crcv.ucf.edu/data/UCF101.php)

    python utils/video2img.py --video-path='the path of ucf101' --save-path='the path for saving images'

step3 -- generating label txt for converted images

    python utils/make_label_txt.py --image-path='the path of saved images'

## Training

In [C3D](https://github.com/facebook/C3D), the input dimensions are 128 × 171 × 16 × 3, in this repo are 128 × 171 × 8 × 3.

During trianing, support three types of length for input clips. check this [script](https://github.com/TianzhongSong/DRN-3D/blob/master/train_DRN-3D.py) for detail.

(1) clip length = 16. I take one sample each two frames.

![16f](https://github.com/TianzhongSong/DRN-3D/blob/master/imgs/16f.jpg)

(2) clip length = 24. I take one sample each three frames.

![24f](https://github.com/TianzhongSong/DRN-3D/blob/master/imgs/24f.jpg)

(3) mixed clip lengths. First, I randomly choose 16 or 24 clip length with 50% probability, then take one sample each two or three frames correspondingly. 

Clips are resized to have a frame size of 128 × 171. On training, I randomly crop input clips into 112×112×8 crops for spatial and temporal jittering. I also horizontally ﬂip them with 50% probability. 

    python train_DRN-3D.py --lr=0.005 --batch-size=16 --drop-rate=0.2 --clip-length=16 --random-length=False --image-path='the path of saved images'

## Results

We use only a single center crop per clip, and pass it through the network to make the clip prediction. For video predictions, we average clip predictions of some clips which are evenly extracted from the video (no overlap).

|clip length    | clip acc      | video acc  |
| :-----------: |:-------------:| :----------:|
| 16            | 58.41%        | 62.80%     |
| 24            | 59.47%        |   -        |
| 16, 24 mixed  | 59.60%        |    -       |

## Reference

[RDN](https://github.com/yulunzhang/RDN)

[DenseNet](https://github.com/liuzhuang13/DenseNet)

[keras-resnet](https://github.com/raghakot/keras-resnet)

[C3D](https://github.com/facebook/C3D)

[c3d-keras](https://github.com/TianzhongSong/C3D-keras)
