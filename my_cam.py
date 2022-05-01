import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from PIL import Image
from PIL import ImageFile
import os 
ImageFile.LOAD_TRUNCATED_IMAGES = True


os.environ["CUDA_VISIBLE_DEVICES"]="3"
print('Is cuda device available ? :', 'Yes' if torch.cuda.is_available() else 'No' )
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #set the device
print('Set device :', device)

#folder to save directorirs
SAVE_DIR = './cam_results'
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

def get_dataloaders():
    img_transform = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=img_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                            shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=img_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64,
                                            shuffle=True, num_workers=8)
    return train_loader, test_loader

# Function to compute CAM 
def return_CAM(feature_conv, weight, class_idx):
    # generate the class -activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx: 
        beforeDot =  feature_conv.reshape((nc, h*w))
        print("weight shape: ", weight.shape, " before dot shape: ", beforeDot.shape)
        # cam = np.matmul(weight[idx], beforeDot)
        cam = weight[idx,] * beforeDot
        cam = cam.mean(axis=0)
        print("Cam shape: ", cam.shape)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

vgg16 = models.vgg16(pretrained=True) 
vgg_mod = nn.Sequential(*list(vgg16.children())[:-1])


class CAM(nn.Module):
    def __init__(self):
        super(CAM,self).__init__() 
        #img = images
        self.fc=nn.Linear(512,10)

    
    def forward(self,x):    
        x = x.view(x.shape[0], 512, 49).mean(2)
        
        # .mean(2).view(1,-1) 
        # x=x.view(-1, 512,7*7).mean(1).view(1,-1)
        x=self.fc(x)
        return  F.softmax(x,dim=1)

model=nn.Sequential(vgg_mod,CAM())


# Defining the traininable parameters 
trainable_parameters = []
for name, p in model.named_parameters():
    if "fc" in name:
        trainable_parameters.append(p)

# Defining the optimizer and the loss function
optimizer = torch.optim.SGD(params=trainable_parameters, lr=0.001, momentum=0.9)  
criterion = nn.CrossEntropyLoss()


# Defining the optimizer and the loss function\
optimizer = torch.optim.SGD(params=trainable_parameters, lr=0.001, momentum=0.9)  
criterion = nn.CrossEntropyLoss()

train_loader, test_loader = get_dataloaders()

# Training of the model 
total_step = len(train_loader)
loss_list = []
acc_list = []

num_epochs = 10

model = model.to(device)

for epoch in range(num_epochs):
    model.train() 
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        # print("i ", i)
        if (i + 100) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))


    loss_list = []
    acc_list = []
    model.eval()
    for i, (images, labels) in enumerate(test_loader):
        if(i > 10):
            break
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        # print(outputs.shape)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

    print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, np.array(loss_list).mean(), np.array(acc_list).mean() * 100))


params = list(CAM().parameters())
weight = np.squeeze(params[-1].data.numpy())



normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.CenterCrop(224),
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

model = model.cpu()

src_imgs = './images/'
# Iterating over all the images in a folder and performing cam on that 
for fname in os.listdir(src_imgs):
    img_pt = os.path.join(src_imgs, fname)
    
    img_pil = Image.open(img_pt)
    img_tensor = preprocess(img_pil)
    img_tensor_ext = torch.zeros(1, img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2])
    img_tensor_ext[0, :, : ,:] = img_tensor
    

    img_variable = Variable(img_tensor.unsqueeze(0))
    img_tensor = img_tensor_ext
    print("img tensor shape: ", img_tensor.shape)

    logit = model(img_tensor)

    h_x = F.softmax(logit, dim=1).data.squeeze()
 
    probs, idx = h_x.sort(0, True)
    probs = probs.detach().numpy()
    idx = idx.numpy()
    
    # predicted_labels.append(idx[0])
    predicted =  train_loader.dataset.classes[idx[0]]
    
    print("Target: " + fname + " | Predicted: " +  predicted) 
 
    features_blobs = vgg_mod(img_variable)
    print("feature blobs shape: ", features_blobs.shape) 
    features_blobs1 = features_blobs.cpu().detach().numpy()
    CAMs = return_CAM(features_blobs1, weight, [idx[0]])

    img = cv2.imread(img_pt)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.5 + img * 0.5
  
    img_save_name = os.path.join(SAVE_DIR, fname + '_cam.png')
    cv2.imwrite(img_save_name, result)   