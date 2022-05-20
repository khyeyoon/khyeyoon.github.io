---
title: "[Paper Review] Contextnet: Improving convolutional neural networks for automatic speech recognition with global context"
last_modified_at: 2022-05-20 00:00:00 -0400
categories: 
  - speech recognition paper
tags:
  - update
toc: true
use_math: true
toc_label: "Getting Started"
---

# Contextnet: Improving convolutional neural networks for automatic speech recognition with global context
> Han, Wei, et al. "Contextnet: Improving convolutional neural networks for automatic speech recognition with global context." arXiv preprint arXiv:2005.03191 (2020).

## Abstract

end-to-end 음성 인식에서 CNN이 좋은 성능을 보여왔지만, 여전히 RNN/Transformer 기반 모델들 보다 성능이 떨어짐

본 논문에서는 이러한 성능 격차를 어떻게 해결할 수 있을지 연구하고, 새로운 CNN-RNN-transducer 구조인 ContextNet 제안

> **ContextNet**은 squeeze-and-excitation 모듈을 추가함으로써 global context 정보를 convolution layers에 통합하는 fully convolutional encoder로 구성

ContextNet의 너비를 조정하는 간단한 scaling method 제안

> 모델 사이즈(widths)는 연산량과 정확도 간 trade-off 관계가 있음

가장 널리 사용되는 Librispeech benchmark(clean/noisy)에서 좋은 성능을 보임

> 외부적인 language model(LM) 없이 2.1%/4.6% WER (word error rate)
>
> LM 추가하여 1.9%/4.1% WER
>
> 10M parameters만으로 2.9%/7.0%

## Introduction

end-to-end (E2E) speech recognition을 위한 CNN 기반 모델들에 대한 관심이 높아지고 있음

- the Jasper model

  외부적인 neural language model을 사용하여 LibriSpeech test-clean에서 2.95% WER로 SOTA와 근접
  
  여러번 쌓은 1D convolutions과 skip connections으로 구성된 deep convolution 기반 encoder
  
  CNN 모델의 정확도와 성능을 개선하기 위해 Depthwise separable convolutions 활용
  
  -> 이러한 CNN 기반 모델의 장점은 parameter efficiency이지만, CNN 기반 모델 중 가장 성능이 좋은 **QuartzNet**은 여전히 RNN/Transformer 보다 성능이 떨어짐
  

RNN/transformer 기반 모델과 CNN 기반 모델의 가장 큰 차이점은 context length에 있음 (문맥을 어느정도 범위까지 고려할 수 있는지)

양방향 RNN 모델의 경우, 이론상으로 하나의 셀은 전체 시퀀스 정보에 접근할 수 있음

Transformer 모델의 경우, attention mechanism은 두개의 서로 다른 타임 스탬프에 존재하는 노드들이 서로의 연산에 참여함

하지만, 제한된 kernel 사이즈를 가진 convolution은 특정 시간에 작은 window 사이즈만을 고려할 수 있음

-> 이러한 문맥은 범위가 작고, global 정보를 담지 못함

본 논문에서 CNN 기반 ASR 모델과 RNN/transformer 기반 모델 간 성능 차이의 주된 원인은 global context의 부족이라고 주장
  
* * *

CNN 모델에서 global context를 향상시키기 위해, the squeeze-and-excitation (SE) layer로부터 영감을 얻어 새로운 CNN 기반 ASR 모델인 **ContextNet** 제안 

- SE layer가 local feature vectors의 시퀀스를 하나의 global context vector로 만듦

- 이러한 context를 다시 각각의 local feature vector로 broadcating 시킴 (global 정보 local로 보내기)

- multiplication을 통해 두가지 정보를 합침

convolution layer 뒤에 SE layer를 위치시켰고, convolution 출력이 global 정보에 접근할 수 있도록 함 ??

실험을 통해, squeeze-and-excitation layers를 ContextNet에 추가시키는 것이 WER 값이 크게 감소시키는 것을 확인함





