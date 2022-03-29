## SketchEdit: Mask-Free Local Image Manipulation with Partial Sketches

[[Paper]](https://arxiv.org/pdf/2111.15078.pdf)&emsp; [[Project Page]](https://zengxianyu.github.io/sketchedit/)&emsp; [[Interactive Demo]](#interactive-demo)&emsp; [[Supplementary Material]](https://maildluteducn-my.sharepoint.com/:b:/g/personal/zengyu_mail_dlut_edu_cn/EeKLBdVAq25GmBAEeoLWNfYBXOBMbrDZHTeT7ApndkrR-w?e=IoHtfo)

<img src="https://github.com/zengxianyu/sketchedit/raw/main/face_gif.gif" width=360>&emsp;&emsp;&emsp;&emsp;&emsp;<img src="https://github.com/zengxianyu/sketchedit/raw/main/image_gif.gif" width=360>&emsp;

## Update

Star this project to get notified.
<p align="center">
  <a href="#"><img src="https://user-images.githubusercontent.com/11280511/150838003-01136487-6671-4cc4-ae4d-bef8e60c605e.png"></a>
</p>

- [x] :new: Uploading custom photos are now enabled in the demo 
- [x] :new: Edit face in a fullbody/upper body photo
<img src="https://github.com/zengxianyu/sketchedit/blob/main/example_body.jpg" width=640>

- [x] :new: Enable general scenes editing in the demo
- [ ] Exposing predicted mask & mask revising? 

## Usage

### environment 

The code only has been tested on Ubuntu 20.04.

GPU is required. 

0. Clone the repo:

```
git clone --single-branch https://github.com/zengxianyu/sketchedit
git submodule init
git submodule update
```

1. Install using the provided docker file ```Dockerfile``` or 

```
conda env create -f environment.yml
```

2. Switch to the installed environment

```
conda activate editline
```


### Testing

0. Download pretrained model

```
chmod +x download/*
./download/download_model.sh
```

1. Use the model trained on CelebAHQ

```
./test_celeb.sh
```

2. Use the model trained on Places

```
./test_places.sh
```

### Interactive demo

[http://47.57.135.203:8001/](http://47.57.135.203:8001/)

## Training code and data
Training code and data is coming soon. Star this project to get notified.

<p align="center">
  <a href="#"><img src="https://user-images.githubusercontent.com/11280511/150838003-01136487-6671-4cc4-ae4d-bef8e60c605e.png"></a>
</p>

## Citation
```
@inproceedings{zeng2022sketchedit,
  title={SketchEdit: Mask-Free Local Image Manipulation with Partial Sketches},
  author={Zeng, Yu and Lin, Zhe and Patel, Vishal M.},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

## Acknowledgments
* DeepFill https://github.com/jiahuiyu/generative_inpainting
* Pix2PixHD https://github.com/NVIDIA/pix2pixHD
* SPADE https://github.com/NVlabs/SPADE
