import torch as th
from torch import nn
import numpy as np
from tqdm import trange
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

sns.set_theme()
device = th.device('cuda' if th.cuda.is_available() else 'cpu')

def load_data():
    train = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())

    train_loader = th.utils.data.DataLoader(train, batch_size=128, shuffle=True)
    test_loader = th.utils.data.DataLoader(test, batch_size=128, shuffle=False) # 测试集不需要打乱顺序

    return train_loader, test_loader

class Adam:
    def __init__(
        self,
        model: nn.Module,
        learning_rate=0.001, # 学习率
        beta1=0.9, # 一阶动量系数
        beta2=0.999, # 二阶动量系数
        epsilon=1e-8 # 防止除0
    ):
        self.model = model
        self.momentum = [th.zeros_like(p) for p in model.parameters()] # 一阶动量
        self.v = [th.zeros_like(p) for p in model.parameters()] # 二阶动量
        self.step_t = 0 # 计数
        self.beta1 = beta1
        self.beta2 = beta2
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        
    def zero_grad(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = th.zeros_like(param.grad)
    
    def step(self):
        self.step_t += 1
        self.momentum = [
            self.beta1 * m + (1 - self.beta1) * p.grad
            for p, m in zip(self.model.parameters(), self.momentum)
        ]
        self.v = [
            self.beta2 * _v + (1-self.beta2) * p.grad
            for p, _v in zip(self.model.parameters(), self.v)
        ]
        for param, m, _v in zip(self.model.parameters(), self.momentum, self.v):
            m_hat = m / (1 - self.beta1 ** self.step_t)
            v_hat = _v / (1 - self.beta2 ** self.step_t)
            param.data = param.data - self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon)


def train(
    model: nn.Module,
    optimizer: Adam,
    device: th.device,
    criterion=nn.NLLLoss(),
    nb_epochs=5000,
    batch_size=128,
):
    testing_accuracy = []
    train_loader, test_loader = load_data()
    for epoch in trange(nb_epochs):
        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.reshape(-1, 28 * 28)
            log_prob = model(X.to(device))
            loss = criterion(log_prob, y.to(device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if epoch % 10 == 0:
            # for testing
            model.train(mode=False) # set no grad
            for batch_idx, (testX, testy) in enumerate(test_loader):
                testX = testX.reshape(-1, 28*28)
                log_prob = model(testX.to(device))
                testing_accuracy.append(
                    (log_prob.argmax(-1) == testy.to(device)).sum().item() / testy.shape[0]
                )
            model.train(mode=True)
    
    return testing_accuracy

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(28*28, 1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 128),
            nn.Dropout(0.4),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=-1)
        )
    
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    labels = [
        "Pytorch Adam",
        "This Implementation"
    ]
    for i, optim in enumerate([th.optim.Adam, Adam]):
        model = Net().to(device)
        optimizer = optim(model, learning_rate=0.001) if i == 1 else optim(model.parameters(), lr=0.001)
        testing_accuracy = train(model, optimizer, device)
        sns.lineplot(x=range(0, 5000, 10), y=testing_accuracy, label=labels[i])
    plt.legend()
    plt.xlabel('Epochs (x100)')
    plt.ylabel('Testing accuracy', fontsize=14)
    plt.savefig('adam.png', bbox_inches='tight', fontsize=14)