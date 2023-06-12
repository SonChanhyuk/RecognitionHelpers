import torch
import torch.nn as nn
from torchvision.models import googlenet, resnet18, resnet50
import torchvision.transforms as transforms
from PIL import Image
from utils._utils import decode, preprocess_audio
import random
import numpy as np

seed = 20236
deterministic = True

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EmotionLSTM(nn.Module):
    def __init__(self, input_shape, hidden_size, n_class):
        super(EmotionLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_shape[1], hidden_size=hidden_size, batch_first=True, dropout = 0.3)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, dropout=0.3)
        self.lstm3 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size//2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size//2, n_class)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = x[:, -1, :] 
        x = self.fc(x)
        return x

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

state = torch.load("model_state_dict/model_resnet50_final.pth",map_location=torch.device('cpu'))
model = resnet50()
num_features = model.fc.in_features
num_classes = 6
model.fc = nn.Linear(num_features, num_classes) 
model.load_state_dict(state)

lstm_state = torch.load("model_state_dict/model_lstm_512_15_3.pth",map_location=torch.device('cpu'))
lstm = EmotionLSTM((352, 15),512,6)
lstm.load_state_dict(lstm_state)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
lstm = lstm.to(device)

def emotion_predict_CNN(path):
    X = preprocess_audio(path)
    image = Image.fromarray(X,"RGB")
    image = transform(image)
    image = image.unsqueeze(dim=0)
    model.eval()
    result = model.forward(image.to(device))
    return decode(torch.argmax(result.squeeze()).item())

def emotion_predict_LSTM(path):
    X = preprocess_audio(path)
    model.eval()
    result = lstm.forward(torch.from_numpy(X).unsqueeze(0).to(device))
    return decode(torch.argmax(result.squeeze()).item())
    
