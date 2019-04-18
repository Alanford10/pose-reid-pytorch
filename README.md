# Pose-based Description of Person Re-identification

## Introduction
A code demo for Zixuan He's undergraduate thesis: A Pose Description of Person Re-identification

## Results

<p align="left">
<img src="https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation/result1.gif", width="720">
</p>


## Require
1. Python 3.6
2. [Pytorch 0.4.0](http://pytorch.org/)
3. pip install pycocotools
4. pip install tensorboardX
5. pip install torch-encoding
6. pip install opencv-python
7. pip install numpy
8. pip install pyyaml(for linux/mac users, for windows users please refer to the [link](https://github.com/philferriere/cocoapi))
9. GPU memory >= 8G

## Trained Models
- Download [models file](https://pan.baidu.com/s/1ayQj_u4PT-YPBHil0v-sVA).
- Put the file **pose_model.pth** under path  **./posereid/models/**
- Put the file **svd_model.pth** under path  **./posereid/models/ft_ResNet50**

## Operation

Run our model by

```bash
python run.py --gpu_ids 0 --svd_val 2 --pose_val 14 --plot_skeleton --src_video_path your_video_path --dst_video_path your_output_video
```
`--gpu_ids` run on specific gpus, e.g. gpu_ids: 0,1,2.

`--plot_skeleton` whether plot the skeleton.

`--svd_val` svd vectors valve.

`--pose_val` pose selection valve. 

`--src_video_path` source video path.

`--dst_video_path` output video path.


## Related repository
- CVPR'17, [SVDNet for Pedestrian Retrieval](https://github.com/layumi/Person_reID_baseline_pytorch)
- CVPR'17, [Realtime Multi-Person Pose Estimation](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation)



## Citation

```
@article{DBLP:journals/corr/SunZDW17,
    title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
    author = {Zhe Cao,
              Tomas Simon,
              Shih-En Wei,
              Yaser Sheikh},
    booktitle = {CVPR},
    year = {2017}
}
```
```
@article{DBLP:journals/corr/SunZDW17,
    title     = {SVDNet for Pedestrian Retrieval},
    author    = {Yifan Sun and
                 Liang Zheng and
                 Weijian Deng and
                 Shengjin Wang},
    booktitle   = {ICCV},
    year      = {2017},
}
```