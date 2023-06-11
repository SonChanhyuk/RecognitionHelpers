# RecognitionHelpers

## speech recognition project


----
#### 최신 작업 현황(lastest working)
* [RNNT_Pretrained.ipynb](RNNT_Pretrained.ipynb)
  * 현재 학습이 nan문제나 잘못된 문자열 패턴으로 수렴되는 문제 때문에 우선적으로 Pretrain된 네트워크를 구현하였음
  * 현재 라이브러리 관련 내용은 https://huggingface.co/eesungkim/stt_kr_conformer_transducer_large 참조
  * 추가 설명은 [NvidiaNemoPretrained.md](NvidiaNemoPretrained.md) 참조
* [RNNT_KoSpeech.ipynb](RNNT_KoSpeech.ipynb)
  * kospeech 라이브러리의 rnn-t를 이용해 학습을 시도함
    * 한국어 데이터셋으로 학습시키면 역시 로스에서 nan문제가 발생함(학습데이터셋 라벨링 문제로 수정중)
* [RNNT_Speech_Recognition_KOR.ipynb](RNNT_Speech_Recognition_KOR.ipynb)
  * torchaudio의 함수를 이용해 구현한 rnn-t로 학습을 시도함
    * 오류를 개선 시키긴 했는데 Decoding알고리즘으로 Inference시 거의 같은 결과를 출력하는 문제를 해결해야함
    
----

### 주요 할일
----
* [x] Blank Tag label로 무조건 수렴하는 문제 해결
* [x] RNN-T 디코딩시 Blank Tag label로 무조건 수렴하는 문제 해결
* [x] RNN-T 구현(API로)
* [ ] RNN-T Online Decoding 구현
* [ ] 한국어 데이터 추가 및 교체후 학습

## RNNT를 구현 및 이용하기 위한 사전지식(Pre-knowledge to implement or use RNNT Net)

---

1. 논문(Paper)
     * [CTC(Connectionist Temporal Classification)](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
     * [Beam Search Decoding](https://aclanthology.org/W17-3207.pdf)
     * [RNN-T](https://arxiv.org/pdf/1211.3711.pdf)

2. 관련링크(Related links)
     * https://ratsgo.github.io/speechbook/docs/neuralam/ctc
     * [Beam Search Decoding](https://amber-chaeeunk.tistory.com/94)
     * [Beam Search for Rnn-t ASR](https://www.youtube.com/watch?v=Siuqi7e9IwU)
     * [Greedy Decoding for RNN-T ASR](https://www.youtube.com/watch?v=dgsDIuJLoJU)
     * [Nvidia NeMo Toolkit](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)
     * [Nvidia NeMo Github](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/models/rnnt_models.py)

# 평가 기법


*   WER(Word Error Rate) : 단어 단위의 틀린 갯수로 평가
    +   공통 평가:
          -   Levenshtein 거리(https://en.wikipedia.org/wiki/Levenshtein_distance)
*   CER (Character Error Rate) : 각 문자 단위의 틀린 갯수로 평가
    +   공통 평가:
          -   Levenshtein 거리(https://en.wikipedia.org/wiki/Levenshtein_distance)
*   RNNTLoss
*   기타 평가 기법:
      +   CTCLoss
