import torch
import torch.nn as nn
import wavencoder

class Wav2VecLSTM(nn.Module):
    def __init__(self, lstm_h, lstm_inp=512):
        super().__init__()
        self.encoder = wavencoder.models.Wav2Vec(pretrained=True)
        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.encoder.feature_extractor.conv_layers[5:].parameters():
            param.requires_grad = True
        
        self.lstm = nn.LSTM(lstm_inp, lstm_h, batch_first=True)
        self.attention = wavencoder.layers.SoftAttention(lstm_h, lstm_h)
    
        self.height_regressor = nn.Linear(lstm_h, 1)
        self.age_regressor = nn.Linear(lstm_h, 1)
        self.gender_classifier = nn.Linear(lstm_h, 1)

    def forward(self, x):
        x = self.encoder(x)
        output, (hidden, _) = self.lstm(x.transpose(1,2))
        attn_output = self.attention(output)

        height = self.height_regressor(attn_output)
        age = self.age_regressor(attn_output)
        gender = self.gender_classifier(attn_output)
        return height, age, gender