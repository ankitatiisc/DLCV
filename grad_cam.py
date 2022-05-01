#Load libraries
import torch
import torch.nn as nn

import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision
import cv2
from PIL import Image

#FOR GPU support
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
print('Is cuda device available ? :', 'Yes' if torch.cuda.is_available() else 'No' )
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #set the device
print('Set device :', DEVICE)

#folder to save directorirs
SAVE_DIR = './cam_results'
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

class ResNet_CAM(nn.Module):
    def __init__(self, net, layer_k):
        super(ResNet_CAM, self).__init__()
        self.resnet = net
        convs = nn.Sequential(*list(net.children())[:-1])
        self.first_part_conv = convs[:layer_k]
        self.second_part_conv = convs[layer_k:]
        self.linear = nn.Sequential(*list(net.children())[-1:])
        
    def forward(self, x):
        x = self.first_part_conv(x)
        x.register_hook(self.activations_hook)
        x = self.second_part_conv(x)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = x.view((1, -1))
        x = self.linear(x)
        return x
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.first_part_conv(x)

def superimpose_heatmap(heatmap, img):
    resized_heatmap = cv2.resize(heatmap.numpy(), (img.shape[2], img.shape[3]))
    resized_heatmap = np.uint8(255 * resized_heatmap)
    resized_heatmap = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)
    inv_norm = transforms.Normalize((-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010), (1/0.2023, 1/0.1994, 1/0.2010))
    superimposed_img = torch.Tensor(cv2.cvtColor(resized_heatmap, cv2.COLOR_BGR2RGB)) * 0.006 #+ inv_norm(img[0]).permute(1,2,0)
    
    return superimposed_img

def get_grad_cam(net, img):
    net.eval()
    pred = net(img)
    pred[:,pred.argmax(dim=1)].backward()
    gradients = net.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = net.get_activations(img).detach()
    for i in range(activations.size(1)):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)
    
    return heatmap #torch.Tensor(superimpose_heatmap(heatmap, img).permute(2,0,1))

baseline_net = torch.hub.load('ecs-vlc/FMix:master', 'preact_resnet18_cifar10_baseline', pretrained=True)


baseline_cam_net = ResNet_CAM(baseline_net, 4)
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([transforms.Resize(224),
   transforms.ToTensor(),
   normalize
])
src_imgs = './images/'

# Iterating over all the images in a folder and performing cam on that 
for fname in os.listdir(src_imgs):
    img_pt = os.path.join(src_imgs, fname)
    
    img_pil = Image.open(img_pt)
    img_tensor = preprocess(img_pil)
    pred = get_grad_cam(baseline_cam_net, img_tensor.unsqueeze(0))
  
    #pdb.set_trace()
    img = cv2.imread(img_pt)
    height, width, _ = img.shape
    
    heatmap = np.uint8(255 * pred.numpy())
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap,(width, height))
    #cv2.applyColorMap(cv2.resize(pred.cpu().numpy().transpose((1,2,0)),(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.5 + img * 0.5
  
    img_save_name = os.path.join(SAVE_DIR, fname + '_grad_cam.png')
    cv2.imwrite(img_save_name, result)   