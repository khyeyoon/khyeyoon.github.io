---
title:  "[Paper Review] Focal Loss for Dense Object Detection paper"
last_modified_at: 2020-07-02 11:50:28 -0400
categories: 
  - Face recognition paper
tags:
  - update
toc: true
toc_label: "Getting Started"
---

# Focal Loss for Dense Object Detection
> Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings of the IEEE international conference on computer vision. 2017.

## Abstract

* detector
> Two stage detector(ex RCNN framework)는 높은 정확도를 보임

> One stage detector는 빠르고 간단하지만 two stage detector에 비해 정확도가 낮음

* One stage detector가 낮은 정확도를 보이는 주된 원인 : class imbalance (fore-ground/back-ground)
* class imbalance 문제를 해결하기 위해 cross entropy loss를 재구성 : novel Focal Loss 제안
> 많은 양의 easy examples을 down-weight 시키면서 적은 양의 hard examples에 더 집중하여 학습할 수 있도록 함
* loss를 test하기 위해 단순한 detector인 RetinaNet을 만들고 실험함
> 기존의 one stage detectors 만큼 빠르고 기존 SOTA인 two stage detector보다 성능은 더 좋음

## Introduction

* Two-stage detectors : R-CNN framework
> first stage : object locations 후보 집합을 생성 (sparse sampling)

> second stage : CNN을 통해 각 후보들을 foreground class들 중 하나 또는 background class로 분류

> two-stage detector는 COCO benchmark challenging에서 top accuracy를 달성

* One-stage detectors : YOLO and SSD
> 단순한 one-stage detetor로 비슷한 성능을 낼 수 있을지에 대한 연구

> Object locations, scale 및 가로 세로 비율이 규칙적이고 밀도 높은 샘플링에 적용됨(dense sampling)

> SOTA인 two-stage methods에 비해 10-40% 정확도(??)를 보이는 더 빠른 detectors

* two-stage detector의 성능을 따라잡는 최초의 one-stage detector 제시
> class imbalance 문제를 해결하는 새로운 loss function 제안

* Two-stage methods에서의 class imbalance 해결법
> R-CNN에서는 two-stage cascade와 sampling heuristics(??)를 통해 해결
> > Proposal stage(ex Selective search, EdgeBoxes, DeepMask, RPN)에서 background를 걸러내면서 object location candidate를 많이 줄임 

> > Second classification stage에서는 fixed foreground-to-background ratio(1:3), online hard example mining(OHEM)와 같은 sampling heuristics를 수행

* One-stage methods의 문제점 
> 이미지에서 너무 많은 양의 candidate를 규칙적으로 sampling함
> > 중요한 정보를 걸러내지 못하고 너무 많은 양을 학습하면서 학습과정이 비효율적임

> sampling heuristic과 유사한 것을 적용할 수는 있지만, 여전히 training 과정에 easy examples(ex background)이 너무 많아서 비효율적임
> > 일반적으로 bootstrapping 또는 hard example mining과 같은 기법을 통해 해결

* Focal Loss 
> 동적으로 scale이 조정되는 cross entropy loss이며, 여기서 scaling factor는 정확한 class에 대한 신뢰도가 증가함에 따라 0으로 감소

> scaling factor는 training 과정에서 자동적으로 easy examples을 down-weight하고 model이 hard examples에 빠르게 집중할 수 있게 함

> class imbalance 문제를 다루던 기존의 방식을 뛰어넘는 것을 실험을 통해 증명

<img src="/assets/img/focal/fig1.PNG" width="50%" height="50%" title="70px" alt="memoryblock">
 
 * RetinaNet
 > Focal loss의 효과를 증명하기 위해 simple one-stage object detector를 만듦
 
 > ResNet-101-FPN backbone으로 구성한 것이 best model
 
 <img src="/assets/img/focal/fig2.PNG" width="50%" height="50%" title="70px" alt="memoryblock">
 
 ## Focal Loss
 
 * One stage detection에서의 class imbalance 문제 해결을 위해 고안된 loss function
 
 * Cross Entropy Loss
 
 <img src="/assets/img/focal/crossentropy.PNG" width="40%" height="40%" title="70px" alt="memoryblock">
 
 * Balanced Cross Entropy
 > class imbalance 문제를 해결하기 위한 흔한 방법은 weighting factor를 도입시키는 것이며, 이때 weighting factor는 class 빈도와 반비례하여 설정되고 cross validation을 설정하기 위한 hyperparameter로 다루어짐
 
 <img src="/assets/img/focal/balancedCE.PNG" width="25%" height="25%" title="70px" alt="memoryblock">
 
 * Focal Loss Definition
 > balanced cross entropy에서 scaling factor는 positive(정답)/negative(오답) examples의 중요성에 균형을 이루지만, easy/hard examples을 구별할 수 없음
 
 > Focal loss : easy examples을 down-weight시키고 hard negatives에 대한 학습에 중점을 둠
 
 > Γ을 증가시킬수록 modulating factor의 영향력이 커짐 (실험에서 2로 설정하는 것이 제일 좋았음)
 
 <img src="/assets/img/focal/focal.PNG" width="30%" height="30%" title="70px" alt="memoryblock">
 
 > α-balanced variant of the focal loss
 
 <img src="/assets/img/focal/balancedfocal.PNG" width="30%" height="30%" title="70px" alt="memoryblock">
 
 * Class Imbalance and Model Initialization
 > binary classification models은 기본적으로 class에 대한 확률값을 동일하게 초기화 (y=-1 일 확률 0.5, y=1 일 확률도 0.5)
 
 > 하지만 사전에 class imbalabce에 대한 정보가 있으면 그에 맞게 초기화 시켜야 학습을 안정적으로 할 수 있음
 
 * Class Imbalance and Two-stage Detectors
 > Two-stage detectors는 class imbalance 문제를 두가지 방식을 통해 해결함
 
 > > 1) a two-stage cascade : object proposal mechanism으로 방대한 양의 data set을 1000~2000개로 줄임
 
 > > 2) biased sampling : 일반적으로 positive와 negative의 비율(ex 1:3)이 minibatch 마다 동일하도록 구성
 
 > focal loss는 위와 같은 mechanisms을 loss function을 통해 one-stage detection에서 해결
 

 
 
 
 
 
 
 












  

















