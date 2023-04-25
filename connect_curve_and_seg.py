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
model3 = torch.load('conv_models/model3.pth').to(device)
model1.eval()
model2.eval()
model3.eval()

x = np.arange(0, 1.02, 0.02)
y = []
z = []
for alpha in x:
    print(alpha)
    sd1 = model1.state_dict()
    sd2 = model2.state_dict()
    for key in sd1:
        sd2[key] = (sd2[key]*alpha + sd1[key]*(1-alpha))
    model_seg = SimpleConv()
    model_seg.load_state_dict(sd2)
    model_seg.eval().to(device)

    if alpha < 0.5:
        sd1 = model1.state_dict()
        sd2 = model3.state_dict()
        alpha *= 2
    else:
        sd1 = model3.state_dict()
        sd2 = model2.state_dict()
        alpha = 2 * (alpha - 0.5)
    for key in sd1:
        sd2[key] = (sd2[key] * alpha + sd1[key] * (1 - alpha))
    model_curve = SimpleConv()
    model_curve.load_state_dict(sd2)
    model_curve.eval().to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    train_transform = transforms.Compose([transforms.ToTensor()])
    train_batch_size = 1000
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=train_batch_size, shuffle=True)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=train_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=train_batch_size, shuffle=False)

    i = 0
    total_loss_seg = 0
    total_loss_curve = 0
    total_corr_curve = 0
    total_corr_seg = 0
    total_seg = 0
    total_curve = 0
    for batch_num, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output_seg = model_seg(data)
        loss_seg = criterion(output_seg, target)
        total_loss_seg += loss_seg.item()
        output_curve = model_curve(data)
        loss_curve = criterion(output_curve, target)
        total_loss_curve += loss_curve.item()
        i += 1

        prediction_seg = torch.max(output_seg, 1)
        total_seg += target.size(0)
        total_corr_seg += np.sum(prediction_seg[1].cpu().numpy() == target.cpu().numpy())

        prediction_curve = torch.max(output_curve, 1)
        total_curve += target.size(0)
        total_corr_curve += np.sum(prediction_curve[1].cpu().numpy() == target.cpu().numpy())
    y.append(total_corr_seg/total_seg)
    z.append(total_corr_curve / total_curve)
    # y.append(total_loss_seg / i)
    # z.append(total_loss_curve / i)

plt.plot(x, y, label="straight")
plt.plot(x, z, label="trained")
plt.ylabel("accuracy")
plt.xlabel("t")
plt.legend(loc="lower left")

plt.savefig('conv_models/both_lines_train_acc_smooth_legend.png')
plt.show()