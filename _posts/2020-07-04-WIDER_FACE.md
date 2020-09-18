---
title:  "WIDER FACE : A Face Detection Benchmark review"
last_modified_at: 2020-07-04 11:50:28 -0400
categories: 
  - Face detection paper
tags:
  - update
toc: true
toc_label: "Getting Started"
---

# WIDER FACE : A Face Detection Benchmark
> Yang, Shuo, et al. "Wider face: A face detection benchmark." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

**Abstract**

* 논문에서 현재의 face detection 성능과 현실에서 필요로 하는 것 사이에는 격차가 있음을 보여줌

* 새로운 face detection dataset을 제시 : WIDER FACE
> 기존의 dataset보다 10배 크고, 많은 annotations을 포함(occlusions, poses, event categories, face bounding boxes)

* 대표적인 detection systems에서 벤치마킹하여 SOTA 성능을 제공하고 많은 scale 변동을 다룰 수 있는 해결책 제시

<img src="/assets/img/wider-face/fig1.PNG" width="40%" height="30%" title="100px" alt="memoryblock">

**Introduction**

* Face detection : 주어진 임의의 이미지에서 이미지내에 사람의 얼굴이 존재하는지 존재하지 않는지 판단하고, 존재한다면 각각의 위치와 범위를 반환

* 최근 연구는 시각적으로 큰 변화가 있는 상황(the unconstrained scenario)에 대한 face detection에 focus를 둠
> public datasets(FDDB, AFW, PASCAL FACE)은 detection이 발전하는데 많은 원동력이 되었지만 아직 더 challenging한 datasets이 필요

* 기존 datasets은 제한된 상황만 다룰 수 있어서 real-world에 적용하기 힘들고 크기가 작음(a few thousand faces)

* contributions
> 1) WIDER FACE : large-scale face detection dataset 제시(393,703개의 face가 labeling된 32,203개의 이미지로 구성)
> > 기존 가장 큰 detection dataset의 10배 크기, 매우 다양한 상황을 다룸 (fig1)

> 2) 제안한 multi-scale two-stage cascade framework에서 WIDER FACE를 사용한 것을 보여줌

> 3) 대표적인 4개의 algorithms에 벤치마킹함
> > 실패원인도 분석

**WIDER FACE Dataset**

* 3.1.Overview
> -WORIDE FACE dataset은 가장 큰 face detection dataset (2019 기준, WIDER dataset에서 추출한 이미지들) 

> -32,203개의 이미지로 이루어져 있고, 총 393,703개의 label이 존재 (scale, pose, occlusion과 같은 다양한 변수 상황을 다룸)

> -60개의 event class로 이루어짐

> -랜덤으로 선택하여 training/validation/test 각각 40/10/50

> * Scenario-Ext 
> > 다른 dataset으로 학습을 하고 WIDER FACE로 testing

> * Scenario-Int
> > WIDER FACE로 학습을 하고 WIDER FACE로 testing

* 3.2.Data Collection

> * Annotation policy

> -인식할 수 있는 모든 얼굴에 bounding box를 labeling 

> -PASCAL dataset과 비슷하게 화질이 좋지 않거나 너무 작은 pixels(10 pixel 이하)은 'Ignore' flag 지정

> -더 나아가 pose(typical,atypical), occlusion level(partial, heavy)에 대한 annotation도 있음

<img src="/assets/img/wider-face/fig2.PNG" width="50%" height="50%" title="100px" alt="memoryblock">

* 3.3.Properties of WIDER FACE

> * Overall

> -WIDER FACE dataset은 기존에 존재하는 dataset보다 더 challenging함

> -three levels of difficulty : Easy, Medium, Hard (EdgeBox의 detection rate를 기준)

<img src="/assets/img/wider-face/fig3.a.PNG" width="50%" height="50%" title="100px" alt="memoryblock">

> * Scale

> three scales : small(between 10-50 pixels), medium(between 50-300 pixels), large(over 300 pixels)

<img src="/assets/img/wider-face/fig3.b.PNG" width="50%" height="50%" title="100px" alt="memoryblock">

> * Occlusion 

> -face detection 성능을 평가하기 위한 중요한 요소

> -three categories : no occlusion, partial occlusion(1-30%), heavy occlusion(over 30%)

> <img src="/assets/img/wider-face/fig3.c.PNG" width="50%" height="50%" title="100px" alt="memoryblock">

> * Pose

>   <img src="/assets/img/wider-face/poselevel.PNG" width="15%" height="15%" title="100px" alt="memoryblock">
> 출처 : https://upload.wikimedia.org/wikipedia/commons/5/54/Flight_dynamics_with_text.png

> two deformation levels : typical, atypical (either the roll or pitch degree is larger than 30-degree or the yaw is larger than 90-degree)

> <img src="/assets/img/wider-face/fig3.d.PNG" width="50%" height="50%" title="100px" alt="memoryblock">

> * Event

> -WIDER FACE는 총 60개의 event를 포함하고 있음

> -각 event가 face detection에 끼치는 영향력을 평가하기 위해 3가지 factor로 특징화 시킴 : scale, occlusion, pose

> -event class에 대한 detection rate를 각 factor 별로 계산하여 순위를 매기고, 그것을 기준으로 3가지로 나눔 : easy(41-60 classes), medium(21-40 classes), hard(1-20 classes)

> <img src="/assets/img/wider-face/fig4.PNG" width="100%" height="100%" title="100px" alt="memoryblock">

**Multi-scale Detection Cascade**
> multi-scale two-stage cascade framework 제안

> <img src="/assets/img/wider-face/fig5.PNG" width="100%" height="100%" title="100px" alt="memoryblock">

> * Multi-scale proposal

> -face classification과 scale classification을 위해 fully convolutional network의 집합(4개의 network)을 공동으로 학습

> -face size별로 4개의 categories로 나눔
 
> -각 network를 경계값 scale size의 image size로 학습(만약 1번째 network라면, image size는 30 pixels)

> positive samples : image patch의 center를 face에 맞추고, scale class labels 지정(table2 참고)

> negative samples : patch를 random으로 crop하고, patch의 IoU 값을 계산하고 0.5 이하이면 scale class 값을 -1로 설정하고 해당 patch는 더 이상 학습과정에 영향을 주지 않음

> * Face detection

> -이전의 stage에서 예측된 windows를 걸러냄

> -이전 stage과 동일한 CNN 구조를 사용하여 face classification과 bounding box regression을 공동으로 학습 (input size도 동일)

> -Face detection : ground truth와 IoU 값이 0.5 이상이면 positive level (아니면 negative)

> => cross-entropy loss 사용

> -Bounding box regression : 각 proposal들 중 ground truth와 가장 가까운 위치로 예측 (proposal이 negative이면 output vector [-1,-1,-1,-1]로 설정)

> => Euclidean loss 사용

**Experimental Results**

* 5.1.Benchmarks
> -대표적인 detector categories의 4가지 분야에서 각각 하나의 algorithm을 선택 : VJ, ACF, DPM, Faceness

> -3.1에서 말한 Scenario-Ext 적용 (외부 dataset에서 학습하고 WIDER FACE에서 test), PASCAL VOC의 evaluation metric 사용

> * Overall

<img src="/assets/img/wider-face/fig6.a.1.PNG" width="30%" height="30%" title="100px" alt="memoryblock">
<img src="/assets/img/wider-face/fig6.a.2.PNG" width="30%" height="30%" title="100px" alt="memoryblock">
<img src="/assets/img/wider-face/fig6.a.3.PNG" width="30%" height="30%" title="100px" alt="memoryblock">

> -3개의 subset에서 Faceness는 다른 방식들을 뛰어넘는 성능을 보임

> -성능은 빠르게 감소하여 모든 방식들이 30 AP보다 낮은 성능을 보임

> -실패 원인을 파악하기 위해 다양한 data subsets에서의 성능을 분석

> * Scale

<img src="/assets/img/wider-face/fig6.b.1.PNG" width="30%" height="30%" title="100px" alt="memoryblock">
<img src="/assets/img/wider-face/fig6.b.2.PNG" width="30%" height="30%" title="100px" alt="memoryblock">
<img src="/assets/img/wider-face/fig6.b.3.PNG" width="30%" height="30%" title="100px" alt="memoryblock">

> -small scale에서 12 AP 보다 좋은 성능을 낸 algorithms이 없음

> => 기존의 face detectors는 small scale을 잘 다루지 못함

> * Occlusion

<img src="/assets/img/wider-face/fig6.c.1.PNG" width="30%" height="30%" title="100px" alt="memoryblock">
<img src="/assets/img/wider-face/fig6.c.2.PNG" width="30%" height="30%" title="100px" alt="memoryblock">
<img src="/assets/img/wider-face/fig6.c.3.PNG" width="30%" height="30%" title="100px" alt="memoryblock">

> -가려진 얼굴을 detection 하는 것은 모든 face detector의 주요 성능 지표임

> -part based models인 Faceness와 DPM은 다른 방식들에 비해 occlusion을 더 잘 다룸

> * Pose

<img src="/assets/img/wider-face/fig6.d.1.PNG" width="30%" height="30%" title="100px" alt="memoryblock">
<img src="/assets/img/wider-face/fig6.d.2.PNG" width="30%" height="30%" title="100px" alt="memoryblock">

* 5.2.WIDER FACE as an Effective Training Source
> ACF와 Faceness에 Scenario-Int 적용 : WIDER FACE로 학습시키고 WIDER FACE testing set으로 test

<img src="/assets/img/wider-face/fig7.PNG" width="90%" height="90%" title="100px" alt="memoryblock">

* 5.3.Evaluation of Multi-scale Detection Cascade

> -제안한 multi-scale cascade algorithm의 효과를 평가

> -ACF-WIDER, Faceness-WIDER models과 다르게 Two-stage CNN 기반 baseline 설립 (multiple face scales을 다룸)

> -WIDER Hard subset에서 Faceness 보다 8.5% AP 향상된 결과를 얻음

> => single network는 multiple scales을 다루는데 어려움이 있음

<img src="/assets/img/wider-face/fig8.PNG" width="90%" height="90%" title="100px" alt="memoryblock">








