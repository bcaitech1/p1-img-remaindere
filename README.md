# P1 Stage Image Classification    
  
AI Tech Boostcamp 내부 주최 경진대회 **13등**  
  
## 개요   
  
사람 얼굴 이미지를 이용하여 인물의 마스크 착용 여부 및 성별, 나이를 분류하는 **Classification 모델 개발**
  
## 코드
  
```
README.md
dataset.py # Dataset 및 Preprocessing, SplitbyProfile(사람)
evaluation.py # Accuracy & F1 score calc
inference.py # load model & Inference
loss.py # Focal, Labelsmoothing, F1, CE, create criterion
model.py # Simple CNN, Efficientnet, Swish
requirements.txt
sample_submission.ipynb # Inference & Save .csv file
train.py # Train
```
  
## 검증 전략  

Baseline Code에서 제공된 dataset중 MaskSplitByProfileDataset을 사용  
  
Valid ratio를 0.2 혹은 0.15, 0.25등으로 다양하게 조정하여 검증 및 학습   

## EDA  
  
한 사람 당 이미지 개수 : 7 마스크 착용 5장, 불량 착용 1장, 미착용 1장  
   -> 근본적으로 마스크 착용에 관한 Class에서 Imbalance 존재  
  
나이 분류에 있어서 청년 / 중년 / 노년의 경우 노년의 데이터가 크게 부족  
![image](https://user-images.githubusercontent.com/48322490/122839983-1f310b00-d334-11eb-8fbd-80569a0a2165.png)  
  
  
## 학습  

공통 : EfficientNet 계열 Model, MADGRAD optimizer, StepLR 사용(LR decay rate(gamma) = 0.5), Normalization 미적용  
  
![image](https://user-images.githubusercontent.com/48322490/122838649-7aadc980-d331-11eb-88db-d91b2f7111ac.png)  

**Architecture**  
1. 기존 Image Classification Task에서 무난한 성능을 보여주었던 EfficientNet을 사용
2. Pretrained=True로 선행 학습된 모델의 fine tuning을 진행
  
**Img Size**  
1. 모델의 빠른 학습을 위해 이미지 크기를 작게 하여 실험 진행  
2. 최종 제출용 모델 학습시에는 resolution 감소로 인한 feature 손실을 줄이기 위해 원본 이미지 사이즈를 사용  
  
**Augmentation**  
Contrast 향상이 얼굴에서 나이를 나타내는 정보를 강화시켜주어 예측에 도움이 된다는 가정 하에 ColorJitter, Histogram Equalization 적용  
그 외로 이미지 내에서 얼굴이 왼쪽이나 오른쪽 한 편으로 치우친 것을 보정하기 위해 RandomVerticalFlip 적용  
  
**Dropout**  
적은 데이터 셋에 큰 백본 모델을 사용하기 위하여 사용  
Dropout이 적용되지 않은 상태에서 B5와 같은 Base 혹은 B7과 같은 Large 모델을 사용할 경우 Overfit으로 인하여 실성능이 좋지 못한 문제를 해결  

**Criterion**  
Class Imbalance 문제 해결을 위해 Class 개수 분석 후 그에 맞는 Weight 부여한 Weighted Cross Entropy Loss, F1 Loss, Focal Loss 에 관해 실험  
안정적으로 학습을 진행할 수 있는 Cross Entropy와 Class Imbalance 문제 해결을 위해 효과적인 F1 loss의 절대값을 고려하여 가중치 부여한 후 Multi Loss로 사용  

## Inference (Ensemble)    
  
Hard Voting 방법 사용  
상기한 6개 학습 모델의 앙상블 결과를 허재섭 캠퍼님의 모델과 최종 앙상블  
가장 큰 차이점은 CenterCrop의 적용 여부, 서로 다른 이미지 영역에서의 Classification이 융합되어 성공적인 Generalization 달성  
Public score (27위)에 비하여 상대적으로 높은 Private score (13위)를 달성할 수 있었음  

## 주요 이슈 및 해결  
  
1. Class Imbalance  
   ->skfold로 Dataset을 구성해 보고, loss 변경, 60세 이상부터 노년으로 분류하지만 57세까지 노년으로 ground truth 변경 등의 방법을 이용  
  
## 발견한 향후 개선점  

1. 아이디어들에 대한 적용 순위를 매기지 않은 FIFO 형태의 작업 scheduling  
   -> 아이디어 발견 후 실험 상황에 맞는 해당 아이디어의 구현 및 적용 순위를 세워야 함  
   
2. 작업 기록 미흡  
   -> 실험 시작 시 해당 실험에 대해 별도로 기록하거나, 1epoch 이후 train 및 validation에 대한 result metric들을 logger를 이용하여 기록  
   
3. 실험 변인 미통제  
   -> 두 개 이상의 변화를 주고 실험할 경우 각 변인에 대한 실험 결과를 알 수 없으므로 각각의 실험 시에 변인을 하나씩 변경하여 실험 진행  
  
  
## Contributors  
:floppy_disk:**[송광원(remaindere)](https://github.com/remaindere)** | :spades:**[허재섭(shjas94)](https://github.com/shjas94)**  
  
## Reference  
**[efficient-pytorch](https://github.com/lukemelas/EfficientNet-PyTorch)**  
   
### data는 저작권 관련 이유로 첨부하지 않았습니다!  
