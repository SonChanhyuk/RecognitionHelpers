# RecognitionHelpers

## speech recognition project


----
#### 최신 작업 현황(lastest working)
* [CTC Net 연습(CTC Net Practice) - Colab](https://colab.research.google.com/drive/1T7oo-t97kyv1gSPZcYg6khkeOjK1t_vy?usp=sharing)
----

### 주요 할일
----
* [ ] Blank Tag label로 무조건 수렴하는 문제 해결
* [ ] RNN-T 구현
* [ ] RNN-T Online Decoding 구현
* [ ] 한국어 데이터 추가 및 교체후 학습

## RNNT를 구현 및 이용하기 위한 사전지식(Pre-knowledge to implement or use RNNT Net)

---

1. 논문(Paper)
  * [CTC(Connectionist Temporal Classification)](https://www.cs.toronto.edu/~graves/icml_2006.pdf])
  * [Beam Search Decoding](https://aclanthology.org/W17-3207.pdf)
  * [RNN-T](https://arxiv.org/pdf/1211.3711.pdf)

2. 관련링크(Related links)
  * https://ratsgo.github.io/speechbook/docs/neuralam/ctc
  * [Beam Search Decoding](https://amber-chaeeunk.tistory.com/94)
  * [Beam Search for Rnn-t ASR](https://www.youtube.com/watch?v=Siuqi7e9IwU)
  * [Greedy Decoding for RNN-T ASR](https://www.youtube.com/watch?v=dgsDIuJLoJU)

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
