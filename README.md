# RecognitionHelpers

## speech emotion recognition

### 진행상황
1. CNN모델 성능이 안좋음
2. wav2vec pretrained된 모델사용 해봤으나 추가 학습이 어려웠음
3. MFCC추출하여 image로 취급하고 resnet, googlenet으로 학습 : 정확도 50%가 최대
4. 데이터셋을 변경하여 학습 : 정확도 80%가 최대
5. 모델 LSTM으로 변경 (keras 썼었으나 모델 불러올때 numpy 버전이 안맞아서 pytorch로 재작성)

---
### 현재 사용 모델 : LSTM

---
### python app.py로 실행가능
