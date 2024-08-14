import numpy as np
import pandas as pd 
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename.endswith('.csv'):
            print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/hubmap-organ-segmentation/train.csv')
test = pd.read_csv('/kaggle/input/hubmap-organ-segmentation/test.csv')

sample_image = cv2.imread('/kaggle/input/hubmap-organ-segmentation/train_images/10703.tiff')
plt.imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
plt.show()

class FDataset(Dataset):
    def __init__(self, dataframe, transform=None, target_size=(512, 512)):
        self.dataframe = dataframe
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_id = self.dataframe.iloc[idx]["id"]
        img_path = f'/kaggle/input/hubmap-organ-segmentation/train_images/{img_id}.tiff'
        img = cv2.imread(img_path)
        
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = cv2.imread(img_path)
        
        rle = self.dataframe.iloc[idx]["rle"]
        height = self.dataframe.iloc[idx]["img_height"]
        width = self.dataframe.iloc[idx]["img_width"]
        mask = rle_to_mask(rle, height, width)
        
        img = cv2.resize(img, self.target_size)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        if self.transform:
            img = self.transform(img)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        
        return img, mask

def rle_to_mask(rle, height, width):
    mask = np.zeros(height * width, dtype=np.uint8)
    rle = list(map(int, rle.split()))
    for i in range(0, len(rle), 2):
        start = rle[i] - 1
        length = rle[i + 1]
        mask[start:start + length] = 1
    return mask.reshape((height, width)).T

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),])

train_dataset = FDataset(dataframe=train, transform=transform, target_size=(512, 512))
test_dataset = FDataset(dataframe=test, transform=transform, target_size=(512,512))
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.cnn1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.cnn2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.cnn3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.cnn4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.cnn5 = nn.Conv2d(128, 3, kernel_size=1)
        
        self._to_linear = None
        self.convs = nn.Sequential(
            self.cnn1,
            nn.ReLU(),
            self.pool,
            self.cnn2,
            nn.ReLU(),
            self.pool,
            self.cnn3,
            nn.ReLU(),
            self.pool,
            self.cnn4,
            nn.ReLU(),
            self.pool,
            self.cnn5, 
        )
        self._get_output_size((3, 512, 512))

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 3)

    def _get_output_size(self, input_shape):
        """ Calculate the size of the flattened output from the convolutional layers. """
        with torch.no_grad():
            x = torch.randn(1, *input_shape)
            x = self.convs(x)
            self._to_linear = int(torch.prod(torch.tensor(x.shape[1:])))
    
    def forward(self, x):
        x = self.convs(x) 
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        x = F.softmax(x, dim=1)
        return x

model = CNNModel()

batch_size = 8
n_iters = 2500
num_epochs = n_iters // (len(train_loader.dataset) // batch_size)
num_epochs = int(num_epochs)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

count = 0
loss_list = []
iteration_list = []
accuracy_list = []

for epoch in range(num_epochs):
    model.train()

    for images, masks in train_loader:

        outputs = model(images)

        masks = masks.squeeze(1)
        masks = masks.long()
        
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        count += 1
        
        if count % 50 == 0:       
            correct = 0
            total = 0
            model.eval()
            with torch.no_grad():
                for images, masks in test_loader:
                    outputs = model(images)

                    _, predicted = torch.max(outputs.data, 1)

                    total += masks.numel()
                    correct += (predicted == masks).sum().item()
            
            accuracy = 100 * correct / float(total)
            
            loss_list.append(loss.item())
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            
            if count % 500 == 0:
                print('Iteration: {}  Loss: {:.4f}  Accuracy: {:.2f} %'.format(count, loss.item(), accuracy))

plt.plot(iteration_list,loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("CNN: Loss vs Number of iteration")
plt.show()

plt.plot(iteration_list,accuracy_list,color = "blue")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("CNN: Accuracy vs Number of iteration")
plt.show()
