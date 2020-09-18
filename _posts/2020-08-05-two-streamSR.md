---
title:  "Two-stream action recognition-oriented video super-resolution review"
last_modified_at: 2020-08-05 00:00:00 -0400
categories: 
  - Video Super-Resolution paper
tags:
  - update
toc: true
toc_label: "Getting Started"
---

# Two-stream action recognition-oriented video super-resolution
> Zhang, Haochen, Dong Liu, and Zhiwei Xiong. "Two-stream action recognition-oriented video super-resolution." Proceedings of the IEEE International Conference on Computer Vision. 2019.

## Abstract

* VSR을 video analytics tasks(ex,action recognition)를 위해 사용할 수 있도록 연구함

* Two-stream network와 같은 CNN 기반 AR 방식들은 low spatial resolution의 video에 직접 적용할 수 없음

> action recognition 이전에 video SR을 수행하면 개선될 수 있음

* two-stream action recognition networks에 맞는 two video SR 방식 제안

> two-stream : the spatial and temporal streams

* spatial-oriented SR(SoSR) network: 움직임이 있는 object에 대한 reconstruction을 강조

> optical-flow를 통해 loss(optical-flow guided weighted mean-squared-error loss)가 계산

* temporal-oriented SR (ToSR) : 연속적인 frames 사이의 temporal continuity를 강조

* 2개의 SOTA recognition networks와 2개의 datasets(UCF101,HWDB51)을 사용하여 실험을 진행

> 제안된 SoSR과 ToSR은 recognition accuracy를 향상시킴

## Introduction

* 최근 CNN을 AR에 적용시키면서 SOTA 성능을 달성함

> classification의 편의를 위해 대부분 fully-connected layers를 사용함

* AR network는 LR video를 직접 model에 적용할 수 없다는 문제점이 존재 (input size가 맞지 않으므로)

> 대부분의 datasets의 크기도 고정적임 (UCF101,HMDB51,Sports-1M)

> 하지만 현실세계에서의 resolution은 고정적이지 않고, low-resolution을 피할 수 없음 (ex, CCTV 영상)

> high-resolution video라도 action을 포함하는 region이 상당히 작은 경우 문제가 있음

* * *

* Resolution problem을 해결하기 위한 방식

1) interpolation을 통한 단순한 re-scaling

2) super-resolution(SR) 적용

> target task에 맞춰진 SR network를 사용하지 않고, 기존의 SR을 그대로 적용함

* * *

* 기존의 SR은 image의 visual quality를 향상 시키기 위해 사용되었지만, 이는 AR에서 필요로 하는 정보에 최적화되어 있지 않음

> 기존의 SR을 그대로 AR에 적용하는 것이 무조건 성능을 향상시키는데 도움이 되는 것은 아님

* visual quality 대신에 recognition quality에 도움을 주는 video SR을 연구

> AR의 전처리로 SR을 사용

* * *

* two-stream AR 방식을 위한 two video SR 방식을 제안

* The spatial stream : Spatial-oriented SR (SoSR)

> 움직이는 object가 recognition에 더 연관성이 있다는 점에 주목하여, SR을 수행할 때 움직이는 object에 대해 집중적으로 수행

> loss function : weighted mean-squared-error(MSE) guided by optical flow

* The temporal stream : Temporal-oriented SR (ToSR)

> 기존 VSR 문제점 : 연속적인 video frame 사이의 temporal discontinuity를 유발함

> > optical flow의 quality를 해치고 recognition 정확도를 떨어뜨림

> 연속적인 frames을 함께 향상시키면서 temporal consistency를 보장

## Action Recognition-Oriented SR

<img src="/assets/img/two-streamSR/fig1.PNG" width="100%" height="100%" title="70px" alt="memoryblock">

* Pipeline

1) LR video sequence를 SR enhancement를 수행할 frames으로 분할

2) ToSR의 결과 영상(frames)에서 optical flow를 계산

3) SoSR의 결과 영상(frames)과 optical flow가 함께 recognition network의 입력으로 들어감

* Action Recognition Network

TSN : two-streams에서 예측된 classification scores의 weighted average를 사용

ST-Resnet : two-streams과 함께 end-to-end 방식으로 융합 sub-network를 훈련

* End-to-End Optimization?

특정 행동 인식 모델(ex,TSN)을 사용하여 SR을 학습시키고 다른 모델(ex,STResnet)을 사용하여 테스트하면 훨씬 더 나쁜 결과를 초래

> SR training을 위한 특정 loss functions을 design 해야함

* * *

### 3.1 Spatial-oriented SR

* 3.1.1 Analysis

spatial stream : 각각의 frames에서 object를 인식하여 recognition을 수행 (= image classification)

SR을 AR에 적용하는 것은 image details이 추가되는 regions이 어떤 부분인지에 따라 도움이 될 수도 있고 해로울 수도 있음

<img src="/assets/img/two-streamSR/fig2.a.PNG" width="60%" height="60%" title="70px" alt="memoryblock">

> class와 관련된 detail(bow,arrow)이 담겨있는 SR frame은 좋은 성능

<img src="/assets/img/two-streamSR/fig2.b.PNG" width="60%" height="60%" title="70px" alt="memoryblock">

> key object보다는 배경정보에 대한 details을 많이 포함하는 SR frame은 성능에 해로움

<img src="/assets/img/two-streamSR/fig2.c.PNG" width="60%" height="60%" title="70px" alt="memoryblock">

> 배경정보를 적게 담고 있고 human(object)에 대한 details을 많이 담고 있는 SR frame은 좋은 성능

* * *

* 3.1.2 Method

* action recognition에 관련성이 높은 image regions을 선택적으로 향상시키는 SR method 제안

> motion 정보를 담고 있는 optical flow에 따라 regions 선택

* 모든 pixel이 동일하게 영향을 끼치던 기존의 MSE loss를 사용하지 않고, optical flow에 기반한 weighted MSE loss를 제안

> 중요한 region의 pixels을 강조

<img src="/assets/img/two-streamSR/eq1.PNG" width="60%" height="60%" title="70px" alt="memoryblock">

> p:pixel index / N:전체 pixel 개수 / u: magnitude of optical flow in horizontal direction, v: magnitude of optical flow in vertical direction

> optical flow는 Flownet 2.0을 사용하여 계산됨

* * *

* 추가적으로, perceptual loss(feature loss, adversarial loss)를 연구

> feature loss : SR image와 HR image 사이의 high-level image features의 차를 최소화하기 위해 사용

> adversarial loss : HR image와 비슷한 SR image를 생성해내기 위해 사용 (in terms of distribution)

* * *

* spatial stream은 image classification과 동일하기 때문에, SISR이 적합할 것이라고 예상

> VDSR, ESRGAN 사용

* 실험을 통해, 최종적인 SoSR은 ESRGAN을 기반으로 학습이 진행됨

<img src="/assets/img/two-streamSR/eq2.PNG" width="60%" height="60%" title="70px" alt="memoryblock">

* * *

### 3.2 Temporal-oriented SR

* 3.2.1 Analysis

temporal stream : temporal 정보를 활용하기 위해 optical flow를 입력으로 취함

> SR이 optical flow의 quality를 높여주어야 함

<img src="/assets/img/two-streamSR/fig3.PNG" width="60%" height="60%" title="70px" alt="memoryblock">

> SR video는 bicubic video에 비해 더 많은 artifacts를 가지고 있어서 recognition 정확도를 떨어뜨림

* * *

* VDSR(SISR)은 frame을 독립적으로 처리하기 때문에 temporal inconsistency를 유발함

> high-quality optical flow를 얻기 위해서는 frames 간 temporal consistency가 보장되야야 함

> > 그렇지 않으면 visible flickering artifacts가 생길 수 있음

<img src="/assets/img/two-streamSR/fig4.PNG" width="60%" height="60%" title="70px" alt="memoryblock">

* * *

* 3.2.2 Method

* 기존의 VSR은 일반적으로 frame을 개별적으로 처리하기 때문에 frames 간 consistency를 보장하기 힘듦

> siamese network : video SR을 학습시키기 위해 연속적인 frame을 함께 복원하는 구조

* * *

* high-quality optical flow를 얻기 위해 SR 프레임 간의 optical flow를 계산하여 HR 프레임 간의 optical flow와 비교하는 것이 간단한 방법이지만, end-to-end 학습을 위해서 optical flow 추정 network가 필요하고, 이는 매우 깊기 때문에 효율적이지 못함

> temporal continuity를 예측하기 위한 warping approach 사용

* * *

* The siamese network training ToSR

<img src="/assets/img/two-streamSR/fig5.PNG" width="60%" height="60%" title="70px" alt="memoryblock">

1) 2개의 연속적인 frames으로 SR을 각각 처리

2) 이전에 HR video에서 계산된 optical flow를 사용하여 SR frame을 warping(bilinear interpolation으로 구현)

3) warped result는 이전 timestamp의 SR와 HR frames과 비교됨

> L_warp-SR / L_warp-HR

<img src="/assets/img/two-streamSR/eq3.PNG" width="60%" height="60%" title="70px" alt="memoryblock">

ToSR은 기존의 어떤 SISR or VSR network로도 구현이 가능함

> VDSR network와 VSR-DUF(with 16 layer) 사용

> temporal stream에서 VDSR은 인접 frame 간 정보가 부족하다는 한계가 있었으나, 반면에 multi-frame SR network는 좋은 성능을 냄













