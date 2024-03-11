import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import os

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(640,640), antialias=True),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])

test_transform = transforms.Compose([
    transforms.Resize((640,640)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=test_transform):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))

        self.img_paths = []
        self.labels = []

        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.img_paths.append(img_path)
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')

        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 160 * 160, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 32 * 160 * 160)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()

train_path = r'C:\Users\Admin\Desktop\Shooter AI\dataset\train'
test_path = r'C:\Users\Admin\Desktop\Shooter AI\dataset\test'

train_dataset = CustomDataset(train_path, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

test_dataset = CustomDataset(test_path, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

if not os.path.isfile('weights_cnn.pt'):
    num_epochs = 15
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f'Evaluating {epoch+1}/{num_epochs}'):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()

        train_loss /= len(train_loader)
        test_loss /= len(test_loader)

        print(f'Epoch {epoch+1}/{num_epochs} Train Loss: {train_loss:.4f} Test Loss: {test_loss:.4f}')

    #save model
    torch.save(model.state_dict(), 'weights_cnn.pt')