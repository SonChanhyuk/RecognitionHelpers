## 개발현황

### 데이터 수집
* 자유대화 음성(일반남여)
  * https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=109
  * 데이터 양이 너무 많아 일부분만 한정하였음
### 데이터셋 경로
```
dataset
└ 자유대화 음성(일반남녀)
  └ Training
    └ label
      └ 일반남여_일반통합..._00002.json
    └ 일반남여_일반통합....
      └ 일반남여_일반통합..._00002.wav
  └ Validation
```
### 데이터 전처리 및 특성 분석
#### 1. Pre-emphasis
S/N(신호대잡음비)을 개선하기 위해 고주파를 강조함. 데이터셋의 특성에 기반함 (일반 녹음기기 사용 등)

pre-emphasis coefficient = 0.97
```
emphasized = librosa.effects.preemphasis(signal)
```
#### 2. Windowing
정상적인 연속 데이터이므로 일반적으로 사용하는 [Hanning window](https://en.wikipedia.org/wiki/Hann_function) 사용
```
# Set window
w = np.hanning(emphasized.size)

# Apply windowing
windowed = w * emphasized
```
#### 3. Fast Fourier Transform
본 [코드](https://github.com/SonChanhyuk/RecognitionHelpers/blob/sonyunseok/Practice.ipynb) 그래프 참고

#### 4. MFCC 추출
MFCC는 오디오 신호에서 추출할 수 있는 feature로, 소리의 고유한 특징을 나타내는 수치임\
MFCC 추출의 일반적인 과정 상 mel spectrum까지 구하는 것이 필요하지만 아직 모델을 구축하기 전이므로 ```librosa.feature.mfcc()```를 사용하고 **이후 세부적인 tuning이 필요할 경우로 미룸**

#### 5. Padding (Optional)
CNN 모델을 사용할경우 해당.\
CNN은 RNN과 달리 제각기 다른 입력 데이터를 읽어들이기 힘듦\
따라서 가장 긴 길이의 데이터를 기준으로 패딩을 채움 -> 이 과정이 문제가 될 수도 있음. 너무 많은 패딩을 채우게 되면 모델의 정확도가 떨어질 수도 있다.\
* 대안
  * 신호 길이의 분포를 통해 적절한 길이 도출 (시도 예정)

## 개발 과제 & 방향
### 데이터셋 라벨 벡터화
각각의 데이터는 문장이 라벨임 (ex. 안녕하세요, 일찍 출근하시네요)\
음성인식의 정확도 향상을 위해 음소(phoneme) 단위로 임베딩하는 알고리즘 내지는 라이브러리가 필요함
### Custom Dataset 구축
pytorch에서 제공하는 데이터셋 형태로 변환. 데이터양이 매우 많기 때문에 변환에 소요되는 시간을 줄이는 방법도 고민해보아야 함
### RNN-T 모델 구축
### 모델 평가

## Link
1. https://blog.naver.com/sexyit_2019/221603292906
2. https://youdaeng-com.tistory.com/5
3. https://dacon.io/competitions/official/235905/codeshare/5201
