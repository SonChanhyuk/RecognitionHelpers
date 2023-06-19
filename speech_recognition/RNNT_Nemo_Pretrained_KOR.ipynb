## Nvidia NeMo Toolkit RNN-Transducer 
---
* Pretrained Network: https://huggingface.co/eesungkim/stt_kr_conformer_transducer_large

## 전체적 구조
---
* RNN Transducer에서 약간 변형된 구조
* 상세 구조는 https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/models/rnnt_models.py 참조

## Encoder 구조
---
* [Conformer](https://arxiv.org/abs/2005.08100)를 기반으로함
  * Conformer란, CNN기반+ Transformer의 self-attention구조를 결합한 형태로 거의 transformer

* 참조 이미지  
  * 이미지 출처: https://velog.io/@sjj995/Conformer-%EB%AA%A8%EB%8D%B8-%EB%A6%AC%EB%B7%B0  와 https://arxiv.org/pdf/2005.08100.pdf
    ![Conformer 구조 이미지](https://velog.velcdn.com/images/sjj995/post/640e4929-603f-4b67-b677-cc703cd2aad5/image.png)

## Decoder 구조
---



## Inference 구조
---
1. 입력: 기본적으로는 **16000HZ** 오디오 파일이어야 한다고 함(resampling 함수등 파일 전처리 필요)
2. 출력: (확률 높은 문장, 모든 문장 후보) 의 tuple 형식
3. Inference 방법: Greedy Search decoding을 사용 (상세는 [여기](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/models/rnnt_models.py#L295)참조)
4. Greedy 동작 개념: https://www.youtube.com/watch?v=dgsDIuJLoJU 
5. 함수 헤더  
   ```python
   #Apache 2.0 license
   #https://github.com/NVIDIA/NeMo/blob/main/LICENSE
   #https://www.apache.org/licenses/LICENSE-2.0
   def transcribe(
        self,
        paths2audio_files: List[str],
        batch_size: int = 4,
        return_hypotheses: bool = False,
        partial_hypothesis: Optional[List['Hypothesis']] = None,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
    ) -> Tuple[List[str], Optional[List['Hypothesis']]]:
        """
        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.
        Args:
            paths2audio_files: (a list) of paths to audio files. \
        Recommended length per file is between 5 and 25 seconds. \
        But it is possible to pass a few hours long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference. \
        Bigger will result in better throughput performance but would use more memory.
            return_hypotheses: (bool) Either return hypotheses or text
        With hypotheses can do some postprocessing like getting timestamp or rescoring
            num_workers: (int) number of workers for DataLoader
            channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`. Uses zero-based indexing.
            augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
            verbose: (bool) whether to display tqdm progress bar
        Returns:
            Returns a tuple of 2 items -
            * A list of greedy transcript texts / Hypothesis
            * An optional list of beam search transcript texts / Hypothesis / NBestHypothesis.
        """
        ```

## 종속성 설치
---
```python
# If you're using Google Colab and not running locally, run this cell.
import os

# Install dependencies
!pip install wget
!apt-get install sox libsndfile1 ffmpeg#for 리눅스
!pip install text-unidecode
!pip install matplotlib>=3.3.2

## Install NeMo
BRANCH = 'main'
!python -m pip install git+https://github.com/NVIDIA/NeMo.git@$BRANCH#egg=nemo_toolkit[all]
```

## 사전 학습된 네트워크 불러오기 및 변수 할당
---
```python
import nemo.collections.asr as nemo_asr
asr_net = nemo_asr.models.ASRModel.from_pretrained("eesungkim/stt_kr_conformer_transducer_large")
```

## 음성 파일 추론하기
---
```python
asr_net.transcribe(['/content/일반남여_일반통합15_M_qudtjdakstp_43_기타_실내_26035.wav'])
#첫번째 리스트는 각각 음성파일 경로명
```
### 사이트에 실려있는 에러 평가
---
| 평가기준 | 링크 |
|--|--|
|CER|7.38%|
|WER|22.73%|

* Conformer기반이라 인식률이 많이 떨어지면, Wav2vec 2.0 Pretrain 구조로 변경할 수 도 있음
  * https://huggingface.co/slplab/wav2vec2_xlsr50k_korean_phoneme_aihub-40m
  * https://huggingface.co/w11wo/wav2vec2-xls-r-300m-korean-lm 

  
## 관련 링크
---
| 주제 | 링크 |
|--|--|
|Conformer 논문| https://arxiv.org/abs/2005.08100|
|Conformer 설명| https://velog.io/@sjj995/Conformer-%EB%AA%A8%EB%8D%B8-%EB%A6%AC%EB%B7%B0 |
|NeMo Toolkit Documention|https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/|
|NeMo Toolkit github|https://github.com/NVIDIA/NeMo|