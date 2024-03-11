import torch
from torchvision import transforms
from PIL import Image
from train import CNN

model = CNN()
classes = ["good", "rotten"]
weights = torch.load('weights_cnn.pt')
model.load_state_dict(weights, strict=False)
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((640,640)),
    transforms.ToTensor(),
])

image = Image.open('img.JPG')
image = transform(image)

output = model(image)

_, predict = torch.max(output, 1)

print(classes[predict.numpy()[0]])