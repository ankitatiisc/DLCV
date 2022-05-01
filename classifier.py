import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms 
import torch.optim as optim 
import torch.nn.functional as F
import os
import numpy as np 
from models import Discriminator, Generator 
import matplotlib.pyplot as plt 
from train_models import get_mnist_test_loader, get_mnist_train_val_loader
# Initializing the device to be gpu
os.environ["CUDA_VISIBLE_DEVICES"]="3"
print('Is cuda device available ? :', 'Yes' if torch.cuda.is_available() else 'No' )
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #set the device
print('Set device :', device)

#folder to save directorirs
SAVE_DIR = './clf_results'
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, (3,3))
        self.conv2 = nn.Conv2d(32, 64, (3,3))
        self.maxpool = nn.MaxPool2d((2,2))
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = self.maxpool(y)
        y = torch.flatten(y, 1)
        y = F.relu(self.fc1(y))
        y = self.fc2(y)

        return y


def train_model():
    num_epochs = 10 
    lr = 0.001
    batch_size = 128

    clf_model = Classifier().to(device)
    
    mnist_train_loader, mnist_val_loader = get_mnist_train_val_loader('./train_data', batch_size=batch_size, shuffle=True,
                    num_workers=4,pin_memory=False, augment=True )
    mnist_test_loader= get_mnist_test_loader('./train_data', batch_size=batch_size, shuffle=True,
                    num_workers=4,pin_memory=False )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(clf_model.parameters(), lr = lr, momentum = 0.9)

    # Training loop 
    for ep in range(0, num_epochs):
        clf_model.train()
        for i, data in enumerate(mnist_train_loader):
            X = data[0].to(device)
            Y = data[1].to(device)
            Y_pred = clf_model(X)
            optimizer.zero_grad()
            loss = criterion(Y_pred, Y)
            loss.backward()
            optimizer.step()

            if (i % 200 == 0):
                output = "[{}][{}/{}] Training loss: {:.3f}".format(ep, i, len(mnist_train_loader), loss.item())
                print(output)

        acc = 0
        losses = 0
        clf_model.eval()
        for id, data in enumerate(mnist_val_loader):
            X = data[0].to(device)
            Y = data[1].to(device)

            Y_pred = clf_model(X)
            loss = criterion(Y_pred, Y)
            losses += loss.item()

            Y_pred_id = torch.argmax(Y_pred, axis=1)
            acc += (Y == Y_pred_id).sum() 

        acc = acc / (len(mnist_val_loader) * batch_size)
        loss = losses / (len(mnist_val_loader))

        ckpt_path = os.path.join(SAVE_DIR, 'model_ckpt_' + str(ep) + '.pth')
        model_dict = {}
        model_dict['state_dict'] = clf_model.state_dict()
        torch.save(model_dict, ckpt_path)

        print("Val metrics - loss: {:.3f}, accuracy: {:.3f}".format(loss, acc))

    acc = 0
    losses = 0
    clf_model.eval()
    for id, data in enumerate(mnist_test_loader):
        X = data[0].to(device)
        Y = data[1].to(device)

        Y_pred = clf_model(X)
        loss = criterion(Y_pred, Y)
        losses += loss.item()

        Y_pred_id = torch.argmax(Y_pred, axis=1)
        acc += (Y == Y_pred_id).sum() 

    acc = acc / (len(mnist_val_loader) * batch_size)
    loss = losses / (len(mnist_val_loader))


    print("Test metrics - loss: {:.3f}, accuracy: {:.3f}".format(loss, acc))

    print("dataset loaded")


# This function will evaluate the model performance on the synthetically generated images 
def evaluate_on_synthetic(classifier_ckpt, gan_checkpoint):
    model = torch.load(gan_checkpoint)
    gen = Generator(nz=64, ndim=64, nc=1)
    gen.load_state_dict(model['gen'])
    gen = gen.to(device)

    clf_w = torch.load(classifier_ckpt)
    clf_model = Classifier()
    clf_model.load_state_dict(clf_w['state_dict']) 
    clf_model.to(device) 
        
    # number of images to sample
    pred_ = []
    N = 100
    iters = 100
    with torch.no_grad():
        for id in range(0, iters):
            z = torch.randn(N, 64, 1, 1, device=device)
            gz = gen(z) # Generated images 

            gz = transforms.Resize(28)(gz)

            y_preds = clf_model(gz)
            y_preds = torch.argmax(y_preds, axis=1).cpu().numpy()
            pred_.append(y_preds)

            if id == 0:
                num_row = 4
                num_col = 5
                # plot images
                images = gz.detach().cpu().numpy()
                fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
                for i in range(num_row * num_col ):
                    ax = axes[i//num_col, i%num_col]
                    ax.imshow(images[i,0], cmap='gray')
                    ax.set_title('Label: {}'.format(y_preds[i]))
                plt.tight_layout()
                plt.savefig(os.path.join(SAVE_DIR, 'predictions_image.png')) 
                plt.close()
        
    pred_ = np.concatenate(pred_)
    
    plt.hist(pred_)
    plt.xlabel('class')
    plt.ylabel('frequency')
    plt.savefig(os.path.join(SAVE_DIR, 'pred_distribution.png')) 
    plt.close()

if __name__ == "__main__":
    train_model()
    #gan_checkpoint = '/data3/ankit/Coursework/DLCV/Assignment_2/ckpts_init/ckpt_9.pth'
    #classifier_checkpoint = '/data3/ankit/Coursework/DLCV/Assignment_2/clf_results/model_ckpt_9.pth'
    #evaluate_on_synthetic(classifier_checkpoint, gan_checkpoint)