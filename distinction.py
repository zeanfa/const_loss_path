import torch
from models import *
import torchvision
import torch.utils.data
from torchvision import transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda')

model1 = torch.load('conv_models/model1.pth').to(device)
model2 = torch.load('conv_models/model2.pth').to(device)
model1.eval()
model2.eval()

criterion = nn.CrossEntropyLoss().to(device)

test_transform = transforms.Compose([transforms.ToTensor()])
batch_size = 10000
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

i = 0
total_loss1 = 0
total_loss2 = 0
pr1 = None
pr2 = None
for batch_num, (data, target) in enumerate(test_loader):
    data, target = data.to(device), target.to(device)
    output1 = model1(data)
    loss1 = criterion(output1, target)
    total_loss1 += loss1.item()

    output2 = model2(data)
    loss2 = criterion(output2, target)
    total_loss2 += loss2.item()

    i += 1

    prediction1 = torch.max(output1, 1)
    pr1 = (prediction1[1].cpu().numpy() == target.cpu().numpy())

    prediction2 = torch.max(output2, 1)
    pr2 = (prediction2[1].cpu().numpy() == target.cpu().numpy())

wrong1 = 0
same_wrong1 = 0
wrong2 = 0
same_wrong2 = 0

for i in range(target.size(0)):
    if pr1[i] == 0:
        wrong1 += 1
        if pr2[i] == 0:
            same_wrong1 += 1
    if pr2[i] == 0:
        wrong2 += 1
        if pr1[i] == 0:
            same_wrong2 += 1

print(wrong1, same_wrong1)
print(wrong2, same_wrong2)
