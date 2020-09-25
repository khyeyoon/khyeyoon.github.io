---
title:  "Deep learning for person re-identification: A survey and outlook review"
last_modified_at: 2020-09-23 00:00:00 -0400
categories: 
  - person re-identification paper
tags:
  - update
toc: true
toc_label: "Getting Started"
---

# Deep learning for person re-identification: A survey and outlook.
> Ye, Mang, et al. "Deep learning for person re-identification: A survey and outlook." arXiv preprint arXiv:2001.04193 (2020).

## Abstract

* person re-identification(Re-ID) : 여러개의 겹치지 않는 cameras에서 찾고자 하는 사람을 찾는 것

> 화면 밖으로 벗어난 사람이 다시 등장하였을 때, 동일 인물인지 알 수 있는 기술

* Re-ID는 closed-world와 open-world 두가지 카테고리로 나누어질 수 있음

> 1. closed-world : deep feature representation learning, deep metric learning, ranking optimization

> > 성능에 한계점이 생기면서, open-world 방식들이 등장 

> 2. open-world : 5가지로 나눠질 수 있음

* 기존 방식들의 장점을 분석함으로써, 강력한 AGW baseline을 디자인

> SOTA를 달성하거나, single/cross-modality Re-ID taske에서 비교할만한 성능을 냄

* person Re-ID를 위한 새로운 evaluation metric mINP 도입

> 올바른 매칭을 찾는 것에 대한 cost를 나타냄

* 마지막으로, 중요하지만 조사되지 않았던 이슈들에 대해 토론함

## Introduction

* person Re-ID : 중복되지 않는 카메라들 사이에서 특정 인물을 찾는 문제

> query(찾고자하는 사람)가 주어지면, 해당 인물이 특정 시간 다른 장소의 카메라에 의해 발견되었는지 알아내는 것이 목적

> query person은 image, video sequence, text description으로 표현됨

* 대학교 캠퍼스, 공원, 도로 등등에서 감시카메라가 증가하고 공공 안전이 중요시되면서, Re-ID의 필요성이 증가하고 많은 발전을 이룸

* * *

* person Re-ID의 challenging 원인 : different view points, low-resolution, illumination changes, unconstrained poses, occlusions, heterogeneous modalities 등등

* deep learning을 통해 많은 발전이 있었지만, 여전히 현실에 적용하기에는 어려움이 있음

### person Re-ID system

<img src="/assets/img/re-identification/fig1.PNG" width="100%" height="100%" title="70px" alt="memoryblock">

1) step 1 : raw data collection

감시카메라로부터 raw video data를 얻는 것으로, 이 data는 복잡하고 노이즈가 있는 background clutter를 포함

2) step 2 : bounding box generation

얻은 데이터에서 사람이 포함되어 있도록 bounding boxes를 추출

> 일반적으로, person detection이나 tracking 알고리즘에 의해 수행됨

3) step 3: training data annotation

cross-camera labels을 annotation

> cross-camera 큰 차이 때문에, Re-ID model 학습을 위해서 필수적인 단계

4) step 4 : model training

구별적이고 강력한 Re-ID model을 학습시키기 위한 제일 중요한 단계

> feature representation learning, distance metric learning, 두 방식의 결합 등등 

5) step 5 : pedestrian retrieval

query와 gallery set이 주어지면, 학습된 Re-ID model을 사용하여 feature representations을 추출하고 query-to-gallery similarity를 계산하여 retrieved ranking list가 얻어짐

## Closed-world person re-identification

* 5-assumptions

1) 사람은 single-modality visible cameras로 촬영됨

2) bounding box로 사람을 나타냄

3) annotation된 데이터세트가 충분하게 구성되어있음

4) annotation은 거의 정확함

5) query person은 gallery set에 반드시 존재함

* * *

## Feature representation learning

<img src="/assets/img/re-identification/fig2.PNG" width="100%" height="100%" title="70px" alt="memoryblock">

### Global feature representation learning

* 각 사람의 이미지에서 global feature vector를 추출하여 Re-ID 수행

* [73] : fine-grained features를 위해 작은 사이즈의 convolutional filters를 사용하는 PersonNet 디자인

<img src="/assets/img/re-identification/personnet.PNG" width="100%" height="100%" title="70px" alt="memoryblock">

* [74] : single-image representation(SIR)와 cross-image representation(CIR)로 구성되어 있는 joint learning framework 제안

> triplet loss를 사용하여 학습을 진행

<img src="/assets/img/re-identification/74.PNG" width="60%" height="60%" title="70px" alt="memoryblock">

* [46] : ID-discriminative Embedding(IDE) model 제안

> 각 identity를 다른 class로 보면서, person Re-ID를 multi-class classification 문제로 해결

* [77] : 다른 scales에서 구별적인 features를 얻기 위해, multi-scale deep representation learning model 제안

> person retrieval을 위한 적절한 scale을 적응적으로 찾음

<img src="/assets/img/re-identification/77.PNG" width="60%" height="60%" title="70px" alt="memoryblock">

* [78] : 픽셀 단위 discriminative features를 얻고, 다양한 자세 변화에 강력해지기 위해 Human semantic parsing 기술을 적용

<img src="/assets/img/re-identification/78.PNG" width="60%" height="60%" title="70px" alt="memoryblock">

* * *

* Attention information

representation learning을 향상시키기 위해 사용

1) For attention within the person image

* [79] : Harmonious Attention CNN(HA-CNN) model 제안 

> misalignment에 강력해지기 위해 soft pixel attention과 hard regional attention을 공동으로 학습

* [80] : 채널 단위 feature response를 조정하기 위한 Fully Attentional Block(FAB) 제안

> FAB는 다른 CNN 구조에도 적용될 수 있음

* [81] : Kronecker Product Matching(KPM) module 제안

> soft warping 기법을 통해 feature maps의 alignment 수행

* [82] : convolutional 구조로 feature maps의 alignment를 수행하는 BraidNet 제안

* [83] : spatial, channel 단위 attention을 수행하는 self-critical reinforcement learning 

* [84] : Mask-Guided Contrastive Attention Model(MGCAM) 제안

> region-level triplet loss로 학습을 하며, background clutter의 영향을 제거함

2) For attention across multiple person images

* [85] : context-aware attentive feature learning 방식 제안

> pair-wise feature alignment와 refinement를 위해, intra-sequence와 inter-sequence 통합

> temporal 정보에 의존적이지 않기 때문에, 이미지 기반 Re-ID에서 multiple images를 하나의 sequence로 만들어 냄

* [86] : attention consistency를 갖는 siamese network를 제안

* [87] : multiple images 간 attention consistency가 고려됨

* [88] : group similarity가 cross-image attention을 하는 또 다른 방법

> 통합된 conditionalrandom field framework에서 image 기반 local, global similarities 모델링

* [89] : spectral feature transformation에도 group similarity 적용

* * *

* Architecture modification

### Local feature representation learning

* part/region aggregated features 학습

> misalignment variations에 강력해짐

* * *

* main trend : 전체 body representation과 local part features를 결합

* [92] : triplet 학습 framework에서 local body part features와 global full body features를 통합시키는 multi-channel parts-aggregated deep convolutional network 제안

<img src="/assets/img/re-identification/92.PNG" width="60%" height="60%" title="70px" alt="memoryblock">

* [93] : multi-scale convolutions을 쌓아 body parts 사이 local context 정보를 캡처

<img src="/assets/img/re-identification/93.PNG" width="100%" height="100%" title="70px" alt="memoryblock">

* [16] : multi-stage feature decomposition, selective tree-structured fusion framework

<img src="/assets/img/re-identification/16.PNG" width="100%" height="100%" title="70px" alt="memoryblock">

* [94] : body를 local regions(parts)로 분해, part-level matching 수행

* [95] : global appearance와 local body part feature maps를 각각 추출하는 two-stream network 제안

> bilinear-pooling layer를 통해 two streams을 결합시킴

<img src="/assets/img/re-identification/95.PNG" width="100%" height="100%" title="70px" alt="memoryblock">

* * *

* background clutter에 대항할 수 있는 part level feature learning이 연구됨

* [96] : Pose driven Deep Convolutional(PCD) model 제안

> 다양한 포즈를 잘 다룰 수 있도록 human part 정보 활용

<img src="/assets/img/re-identification/96.PNG" width="60%" height="60%" title="70px" alt="memoryblock">

* [97] : attention-aware compositional network로 원하지 않는 배경 feature를 가리기 위한 pose-guided part attention module 개발

> 또한, part-level features를 모아줌

<img src="/assets/img/re-identification/97.PNG" width="60%" height="60%" title="70px" alt="memoryblock">

* [98] : person-region guided pooling deep neural network로 background bias를 해결하기 위해, human parsing 활용

<img src="/assets/img/re-identification/98.PNG" width="60%" height="60%" title="70px" alt="memoryblock">

* [99] : two-stream network, full image representation learning + densely semantically-aligned part feature learning

<img src="/assets/img/re-identification/99.PNG" width="100%" height="100%" title="70px" alt="memoryblock">

* [100] : human parts(사람의 신체)와 non-human parts(가방,소지품)가 alignment 됨

<img src="/assets/img/re-identification/100.PNG" width="100%" height="100%" title="70px" alt="memoryblock">

* * *













