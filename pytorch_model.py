import torch 
import torch.nn as nn
from torchvision import transforms

import cv2

class PytorchTrainer:
    def __init__(self, learning_rate=0.0001):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = PytorchModel(2).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, observation_batch, action_batch):
        observation_batch = torch.Tensor(observation_batch).to(self.device).permute(0, 3, 1, 2)
        action_batch = torch.Tensor(action_batch).to(self.device)

        output = self.model(observation_batch)
        loss = self.criterion(output.squeeze(), action_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def save(self):
        torch.save(self.model.state_dict(), 'trained_models/pytorch_convnet.pth')
    
    def load(self):
        self.model.to(torch.device('cpu'))
        self.model.load_state_dict(torch.load("trained_models/pytorch_convnet.pth", map_location='cpu'))
        return self.model

class PytorchModel(nn.Module):
    def __init__(self, output_size=1):
        super(PytorchModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1))

        self.fc = nn.Linear(64 * 15 * 20, output_size)
        
    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

    def predict(self, input):
        input = cv2.resize(input, (80, 60))

        input = torch.Tensor(input).permute(2, 0, 1).unsqueeze(0)
        #print(input.device)
        return self.forward(input).squeeze().detach().numpy()
