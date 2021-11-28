---
layout: page
title: "SketchEdit"
permalink: /
---
## SketchEdit: Mask-Free Local Image Manipulation with Partial Sketches

Yu Zeng<sup>1</sup>, Zhe Lin<sup>2</sup>, Vishal M. Patel<sup>1</sup>

<sup>1</sup>Johns Hopkins University, <sup>2</sup>Adobe Research

[[Paper]]()&emsp; [[Code]](https://github.com/zengxianyu/sketchedit)&emsp; [[Demo]](http://47.57.135.203:8001/) &emsp; [[Results]](#results)&emsp; [[Supplementary Material]]()

<img src="face_gif.gif" width=360>&emsp;&emsp;&emsp;&emsp;&emsp;<img src="image_gif.gif" width=360>&emsp;

![](teaser.jpg)

## Abstract
Sketch-based image manipulation is an interactive image editing task to modify an image based on input sketches from users. Existing methods typically formulate this task as a conditional inpainting problem, which requires users to draw an extra mask indicating the region to modify in addition to sketches. The masked regions are regarded as holes and filled by an inpainting model conditioned on the sketch. With this formulation, paired training data can be easily obtained by randomly creating masks and extracting edges or contours. Although this setup simplifies data preparation and model design, it complicates user interaction and discards useful information in masked regions. To this end, we investigate a new paradigm of sketch-based image manipulation: mask-free local image manipulation, which  only requires sketch inputs from users and utilizes the entire original image. Given an image and sketch, our model automatically predicts the target modification region and encodes it into a structure agnostic style vector. A generator then synthesizes the new image content based on the style vector and sketch. The manipulated image is finally produced by blending the generator output into the modification region of the original image. Our model can be trained in a self-supervised fashion by learning the reconstruction of an image region from the style vector and sketch. The proposed method offers simpler and more intuitive user workflows for sketch-based image manipulation and provides better results than previous approaches. More results, code and interactive demo will be available at \url{https://zengxianyu.github.io/sketchedit}. 

## Results
![](caption.png)
![](image_supp.jpg)
![](face_supp.jpg)

## Interactive Demo

[http://47.57.135.203:8001/](http://47.57.135.203:8001/)
