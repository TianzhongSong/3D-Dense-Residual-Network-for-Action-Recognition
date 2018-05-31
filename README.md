# DRN-3D

3D residual dense network for action recognition 


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

    python train_DRN-3D.py --lr=0.005 --batch-size=16 --drop-rate=0.2 --clip-length=16 --image-path='the path of saved images'

## Todo

Update network detials and results

## Reference

[RDN](https://github.com/yulunzhang/RDN)

[DenseNet](https://github.com/liuzhuang13/DenseNet)

[keras-resnet](https://github.com/raghakot/keras-resnet)

[C3D](https://github.com/facebook/C3D)

[c3d-keras](https://github.com/TianzhongSong/C3D-keras)
