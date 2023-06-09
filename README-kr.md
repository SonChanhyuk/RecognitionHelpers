[![en](https://img.shields.io/badge/lang-en-red.svg)](./README.md)
[![kr](https://img.shields.io/badge/lang-kr-yellow.svg)](./README-kr.md)

# **RecognitionHelpers**

## **구현목표**
청각 장애인에게 도움이 될 수 있는 서비스 - 버튼을 눌러 음성인식을 시작하여 음성인식 결과와 발화자의 감정 예측결과를 보여주는 서비스

---
## **음성인식**
### Datasets used
- Emformer RNN-T 네트워크를 학습시키기 위해, AIHub의 [자유대화 음성(일반남여)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=109)' 데이터셋을 사용했습니다.

### Inference Process
1. 이미 NeMo Toolkit 라이브러리에서 구현된 'transcribe' 함수를 호출합니다.
2. transcribe 함수는 오디오 파일 경로 이름 문자열을 포함하는 리스트 객체를 인자로 받습니다.
3. transcribe 함수 내에서는 각 파일 이름이 순서대로 접근되고 각 오디오 파일이 로드된 다음 MFCC와 같은 특성으로 변환됩니다.
4. 이러한 MFCC 특성들은 RNN-Transducer의 인코더 부분으로 입력되어 특정 시간 단계 세그먼트에 대한 벡터를 얻습니다.
5. 이 벡터들과 초기 RNN 토큰, 빈 토큰을 이용하여 각 토큰을 하나씩 추론하는 탐욕적 디코딩 알고리즘, 즉 Next Token Prediction을 수행하며 다음 오디오 특성의 토큰을 예측합니다.
6. 각 시간 단계에 대한 토큰 값을 추출한 후, 이 값들은 하나의 벡터로 모아집니다. 이 벡터는 단어 임베딩 유형에 대해 학습된 SentencePiece 라이브러리를 사용하여 단어나 문자로 다시 변환됩니다. 이 과정은 연결(concatenate) 연산을 통해 이루어집니다.
7. 마지막으로, 함수는 가장 가능성이 높은 문장 후보와 다양한 다른 문장 추론 후보를 반환합니다.
그리고 이후에 음성 인식 결과와 감정 인식 결과를 연결합니다.
![Process](https://i.imgur.com/XXPvkk0.png)


## Model
* 우리는 torchaudio 라이브러리의 [Emformer-RNN-T](https://pytorch.org/audio/stable/_modules/torchaudio/models/rnnt.html#emformer_rnnt_base)를 학습시키려 했지만, 결과가 항상 편향적으로 잘못된 하나 또는 몇 가지 패턴의 시퀀스로 수렴하는 추론 문제가 있었습니다(우리는 이것이 한국어에 대한 충분하지 않은 훈련, 충분하지 않은 은닉 특징의 수 때문이라고 생각합니다).
* 이러한 진행 상황은 [여기](speech_recognition/RNNT_Emformer_KOR.ipynb)에서 볼 수 있습니다. 
* 그래서 우리는 Emformer RNN-T가 더 많은 훈련이 필요하다고 판단하여 실시간 음성 인식을 위해 RNN transducer 구조를 사용하기로 결정하였습니다.

| RNN-T Base     | RNN-T Encoder  |
| -------------- | -------------- |
| <img src="https://i.imgur.com/45quMCC.png" style="display:inline;" width="300em" height="auto">| ![Process](https://velog.velcdn.com/images/sjj995/post/640e4929-603f-4b67-b677-cc703cd2aad5/image.png)   |
---
## **감정인식**
### 사용된 dataset
- 감정 분류를 위한 대화 음성 데이터셋(한국어) - https://aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&dataSetSn=263&aihubDataSe=extrldata
- RAVDESS - https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
- TESS - https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
- CREMA-D - https://www.kaggle.com/datasets/ejlok1/cremad
- SAVEE - https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee
  
### 개발 진행 과정
1. 한국어 오디오 데이터셋으로 Mel-spectrogram을 추출하여 기본적인 CNN으로 학습 -> 성능이 안좋음
2. 모델을 wav2vec pre-trained된 모델로 변경 -> 추가 학습이 어렵고 모델이 굉장히 무거움
3. MFCC추출하여 3x224x224 image로 만든후 resnet, googlenet으로 학습 : 정확도 50%가 최대
4. 데이터셋을 영어 데이터셋으로 변경하여 학습 : 정확도 최대 80%
5. 모델 전처리 과정에 librosa.effects.normalize, ZCR, RMS추가
6. 모델을 LSTM기반 모델로 변경 -> 정확도 test, validation dataset에서 최대 80% 이상, 그러나 한국어 데이터에 대해서는 매우 안좋은 결과가 나옴

### 모델
구체적인 학습과정은 ./emotion_recognition 폴더의 speech-emotion-recognition.ipynb,resnet_newdataset.ipynb 에서 확인 가능합니다.
실제로 학습을 해보고 싶다면 ./emotion_recognition 폴더의 train.ipynb을 참고해주세요.
학습시 변경 가능한 파라미터는 다음과 같습니다.
- preprocess_audio의 top_db를 바꾸면 오디오 전후에서 노이즈로 인식하는 소리의 크기를 설정할 수 있습니다.
- preprocess_audio의 수치를 바꾸어 읽을 오디오 데이터의 길이를 정할 수 있습니다. 현재 기본값은 180000입니다.
- 전처리 코드에서 X의 구성을 바꾸어도(다른 feature를 사용하여도) 동일하게 코드가 사용가능합니다.
- EmotionLSTM(X.shape[1:3],hidden,n_class)에서 hidden을 통해 LSTM의 히든레이어 수가 변경됩니다.
- 만약 label의 개수가 6개가 아닌 다른 데이터 셋을 사용한다면 n_class를 수정해주시면 됩니다.
- CNN모델의 경우 resnet18, resnet50, googlenet을 사용해보았지만 다른 CNN 모델을 사용하셔도 됩니다.
![emotion_model](./image/emotion_model_image.png)

### 문제점
- dB이 커질수록 예측 결과가 happy, angry로 귀결됨
- 입력 음성데이터의 db을 일반화하는 전처리과정이 필요함 (이러한 효과를 노리고 librosa의 normalize를 사용했으나 효과가 없었음)
---

## 모델 성능
WER (Word Error Rate) : 인식된 단어들과 정답 단어들 사이의 단어 오류 비율  
CER (text Character Error Rate) : 인식된 문자열과 실제 문자열사이의 문자 오류 비율  
Accuracy  
F1-score

|     | 음성인식 |
|-----|---------|
| WER | 22.73%  |
| SER | 7.38%   |

|     | 감정인식 |
|-----|---------|
| acc | 81.25%  |
| f1  | 0.8148  |


## **demo 사용방법**
### 설치(windows)
pytorch 설치
~~~
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
~~~
음성인식을 위한 Nvidia의 NeMo toolkit설치
NeMo toolkit설치시에 윈도우에서는 C++ 빌드 툴이 없으면 설치가 제대로 되지않습니다. visual c++ build를 설치하셔야 합니다.
c++ build가 있어도 pynini는 제대로 설치가 안되기 때문에 아래와 같이 따로 설치를 하셔야합니다.
~~~
conda install -c conda-forge pynini
pip install Cython
pip install "nemo_toolkit[all]"
~~~
demo app 실행을 위한 라이브러리 설치
~~~
pip install SpeechRecognition
pip install pyaudio
~~~
앱 실행 - 로딩이 다되고 나서 127.0.0.1:5000에 접속하시면 됩니다. ___첫 접속에서는 화면이 뜨지 않는채로 음성인식을 받습니다.___
~~~
python app.py
~~~
![result](./image/run_result.jpg)

---
## 디렉토리 구성
- speech_recognition : 음성인식 연습 및 학습용 폴더입니다.
- emotion_recognition : 감정인식 연습 및 학습용 폴더입니다.
- model_state_dict : 학습된 모델 파라미터입니다.
- templates : demo app에서 보여줄 HTML파일입니다.
- utils : emotion_model.py에서 사용할 함수들이 저장되어 있습니다.
- app.py : 데모앱 실행 파일입니다.
- emotion_model.py : 감정인식 모델이 저장되어 있습니다.

## 연락처
손찬혁 - thscksgur0903@gmail.com  
황세현 - imscs21@naver.com  
손윤석 - suryblue@naver.com  
Jungmann, Matthew - wmjungmann@gmail.com  
Sun, Yuekun - maoruoxi520@gmail.com  