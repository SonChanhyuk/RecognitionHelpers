import speech_recognition as sr
import torch
from flask import Flask, render_template, request, redirect
from emotion_model import emotion_predict_LSTM, emotion_predict_CNN
import nemo.collections.asr as nemo_asr
asr_net = nemo_asr.models.ASRModel.from_pretrained("eesungkim/stt_kr_conformer_transducer_large")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
asr_net = asr_net.to(device)

PATH = "input.wav"

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    r=sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        transcript = ""
        audio=r.listen(source)
        with open(PATH,"wb") as f:
            f.write(audio.get_wav_data())
        try:
            #transcript=r.recognize_google(audio, language="ko-KR")
            transcript = asr_net.transcribe([PATH])[0][0]
            emotion = emotion_predict_LSTM(PATH)
            transcript = emotion + " - " + transcript
            print("Speech Recognition : "+transcript)
        except sr.UnknownValueError:
            print("Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("; {0}".format(e))
        return render_template('./index.html', transcript=transcript)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)