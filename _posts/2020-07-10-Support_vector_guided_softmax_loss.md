---
title:  "[Paper Review] Support Vector Guided Softmax Loss for Face Recognition paper"
last_modified_at: 2020-07-10 11:50:28 -0400
categories: 
  - Face recognition paper
tags:
  - update
toc: true
toc_label: "Getting Started"
---

# Support Vector Guided Softmax Loss for Face Recognition
>Wang, Xiaobo, et al. "Support vector guided softmax loss for face recognition." arXiv preprint arXiv:1812.11317 (2018).

**Abstract**

 * CNN을 통해 face recognition은 상당한 발전이 있었지만, feature discrimination 아직 challenge로 남아있음

 * Learn discriminative features

> * mining-based strategies : hard example mining and focal loss

> - 중요한 정보에 focus를 두고 학습

> * margin-based loss functions : angular, additive and additive angular margins

> - ground truth class로부터 feature margin이 커지도록 학습
 
 => 문제점 : hard examples에 대한 기준이 애매모호하고, 다른 class로부터 구별해내는 능력이 부족함
 
 * novel loss function인 SV-softmax(support vector guided softmax loss) 제안
 
 > - adaptively emphasizes the mis-classified points (support vectors) to guide the discriminative features learning
 
 > - 다른 class로부터 구별하는 능력이 있고, hard examples의 애매모호함을 제거할 수 있음
 
 > - 최초로 mining-based 방식과 margin-based 방식의 이점을 하나의 framework로 통합하는 시도를 함
 
 > - 몇몇 benchmarks에서 진행한 실험에서 SV-softmax의 효과를 보여줌
 
  **Introduction**
 
 * Face recognition은 두가지 categories를 포함 
 
 - Face identification : 주어진 face를 특정 identity로 분류

 - Face verification : 한 쌍의 face가 동일한 identity인지 판단
 
 => Face recognition에는 아직 많은 challenge가 존재함 (특히, MegaFace Challenge, Trillion Pairs Challenge와 같은 large-scale test datasets에서)
 
 * 최근 face recognition task에 CNN이 사용됨
 
 * deep models을 학습 시키기 위해 CNN은 일반적으로 classification loss나 metric learning loss 또는 두가지를 모두 사용
 
 => face recognition task에 기존 loss function을 사용하면 계산비용, 성능 면에서 한계가 있음
 
 * face feature는 서로 다른 클래스간 차별성이 존재하지만 기존 loss는 차별성을 잘 다루지 못함
 
 => mining-based loss functions, margin-based loss functions 등장
 
 * Mining-based loss functions (focus on optimizing hard examples)
 
 1) HM-Softmax (hard mining softmax) : minibatch를 high-loss를 사용하여 구성하면서 feature discrimination을 증가시킴
 
 > hard examples의 비율이 경험적으로 결정되고 easy examples이 완전히 제거됨 
 
 2) F-softmax (Focal loss) : 적은 hard examples에 focus를 두고 학습
 
 > simple hard mining softmax보다 좋은 결과
 
 3) Yuan et al.[39] : ensemble 기법을 이용하여 다양한 hard level을 학습
 
 => drawback of mining-based loss : hard examples에 대한 기준이 명확하지 않고 경험적으로 선정됨 
 
 * Margin-based loss functions (increasing the feature margin between different classes)
 
 1) Wen et al.[37] : class 내 compactness를 향상시키기 위해 각 identity의 centers를 학습하기 위한 center loss
 
 2) Wang et al.[32] and Ranjan et al.[19] : class 내 변화량을 줄이기 위해 잘 분리된 samples에 높은 gradient를 생성하고 softmax의 정도를 조절하기 위해 scale parameter 사용
 
 3) A-Softmax (angular margin) : ground truth class와 다른 classes 간 변화량이 커지도록 학습
 
 > 학습 과정이 불안정하고 optimal parameter를 찾기 어려움 -> 학습의 안정성을 향상시키기 위한 방식들이 연구됨 (AM-Softmax)
 
 4) AM-Softmax (additive margin loss) : optimization 과정을 안정화 시키고, 성능을 향상
 
 5) Arc-Softmax (additive angular margin loss) : 더 명확한 geometric interpretation
 
 => drawback of margin-based loss : 오직 ground truth class 관점에서의 margin(self-motivation)을 크게 만들고 non-ground truth class 관점에서의 margin(other motivation)은 무시
 
 * SV-Softmax loss
 
 > - mining-based losses와 margin-based losses 간 격차를 줄이고 의미론적으로 그들을 하나의 framework로 통합하기 위해 적응적으로 informative support vectors를 강조
 
 > - hard examples의 모호함을 제거하고, support vector에 집중하면서 다른 classes로부터의 차별성도 학습
 
 > - 기존 방식(mining,margin loss)과 SV-Softmax의 관계를 분석하고 더 나아가 feature discrimination을 향상시키기 위한 SVX-Softmax를 개발
 
 > - benchmarks(LFW, MegaFace Challenge and Trillion Pairs Challenge)에서 실험을 진행하여 SV-Softmax의 우수함을 증명
 
 **Preliminary Knowledge**
 
 * Softmax
 
 > - softmax loss : the pipeline combination of the last fully connected layer, the softmax function and the cross-entropy loss
 
 <img src="/assets/img/SV-Softmax/eq1.PNG" width="70%" height="70%" title="70px" alt="memoryblock">
 
 > - ground truth의 weight와 input feature vector x 간 angle을 통해 계산한 cosine similarity를 이용
 
 > - softmax loss를 통해 학습된 features는 face recognition에 대해 차별적이기보다는 분리될 수 있는 경향이 있음 (??)
 
 * Mining-based Softmax
 
 > - informative examples에 focus를 두고 학습을 시키면서 일반적으로 더 discriminative features 생성
 
 > - 최근 연구들은 hard examples을 loss value나 model complexity를 기준으로 선정
 
 <img src="/assets/img/SV-Softmax/eq2.PNG" width="70%" height="70%" title="70px" alt="memoryblock">
 
 > - ground truth일 확률과 indicator function을 이용
 
 > - hard examples에 대한 정의가 애매모호하고, 그것이 성능에 영향을 끼침(sensitive performance)
 
 * Margin-based Softmax
 
 <img src="/assets/img/SV-Softmax/eq3.PNG" width="70%" height="70%" title="70px" alt="memoryblock">
 
 > - margin function 사용
 
 > - 오직 ground truth class y의 관점에서만 적용하여 non-ground truth class의 영향력은 무시
 
 **Problem Formulation**
 
 * 3.1 Naive Mining-Margin Softmax Loss
 
 <img src="/assets/img/SV-Softmax/eq4.PNG" width="70%" height="70%" title="70px" alt="memoryblock">
 
 > eq4와 같이 단순히 결합하면 근본적인 문제가 해결되지 않음
 
 * 3.2 Support Vector Guided Softmax Loss
 
 > - 기존의 hard examples에 대한 명확한 기준이 없는 문제를 해결하기 위해 support vectors를 사용
 
 > informative features에 focus를 두고 학습을 시키는 더 정교한 방법
 
 > - binary mask 사용 : 적응적으로 sample이 선택될지 안될지 결정
 
 <img src="/assets/img/SV-Softmax/eq5.PNG" width="70%" height="70%" title="70px" alt="memoryblock"> 
 
  <img src="/assets/img/SV-Softmax/fig1.PNG" width="70%" height="70%" title="70px" alt="memoryblock"> 
 
 > mis-classified samples이 일시적으로 강조되는 것을 확인할 수 있고, 이를 통해 hard samples이 명확하게 정의되고 그러한 sparse set에 focus를 둠
 
 > - SV-Softmax loss
 
 <img src="/assets/img/SV-Softmax/eq6.PNG" width="70%" height="70%" title="70px" alt="memoryblock"> 
 
 > t는 preset hyperparameter, h는 indicator function (t가 1이면, original softmax와 동일)
 
 * 3.2.1 Relation to Mining-based Softmax Losses
 
 <img src="/assets/img/SV-Softmax/fig2.PNG" width="70%" height="70%" title="70px" alt="memoryblock"> 
 
 > Mining-based loss : hard samples(x1)과 easy samples(x2)을 별도로 re-weight 시킴 
 
 > SV-Softmax loss : 의미론적으로 decision boundary를 따라 hard examples을 정의(support vectors)하고 support vector x1에 대한 확률을 감소시킴
 
 * 3.2.2 Relation to Margin-based Softmax Loss
 
 <img src="/assets/img/SV-Softmax/fig4.PNG" width="70%" height="70%" title="70px" alt="memoryblock"> 
 
 > Margin-based loss : ground truth class의 관점에서의 margin function 도입
 
 > SV-Softmax loss : other non-ground truth class의 관점에서의 margin function 도입(class specific margins)
 
 > - Pipeline of SV-Softmax loss and its relations to the existing mining-based and margin-based losses
 
 <img src="/assets/img/SV-Softmax/fig3.PNG" width="90%" height="90%" title="70px" alt="memoryblock"> 
 
 * 3.2.3 SV-X-Softmax
 
 > SV-Softmax loss는 의미론적으로 mining-based loss와 margin-based loss를 하나의 framework로 융합시킴
 
 > mining range를 증가시키기 위해 margin-based decision boundaries를 적용시킴 (SV-X-Softmax)
 
 <img src="/assets/img/SV-Softmax/eq12.PNG" width="70%" height="70%" title="70px" alt="memoryblock"> 
 
 > ground truth로부터의 self-motivation과 other classes로부터의 other-motivation을 하나의 framework로 통합시키면서 feature margin을 크게 하였고, mining-based losses을 사용하면서 의미론적으로 mining range를 크게 함
 
  <img src="/assets/img/SV-Softmax/eq13.PNG" width="70%" height="70%" title="70px" alt="memoryblock">
  
  <img src="/assets/img/SV-Softmax/fig5.PNG" width="70%" height="70%" title="70px" alt="memoryblock">
 
 
 
 
 
 
 
 
 
 
  
 
 

 
  
 
 
 
 
 
 
 
 



