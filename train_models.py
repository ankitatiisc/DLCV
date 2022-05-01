# This module will perform training of DCGAN model on MNIST dataset 


import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms 
import torch.utils.data
import torch.optim as optim
from models import Discriminator, Generator 
import torchvision.utils as vutils 
import matplotlib.pyplot as plt 
import cv2
import numpy as np
import os
from torchvision import datasets

# Initializing the device to be gpu
os.environ["CUDA_VISIBLE_DEVICES"]="3"
print('Is cuda device available ? :', 'Yes' if torch.cuda.is_available() else 'No' )
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #set the device
print('Set device :', device)

# Weights initialization for the given models 
def weights_init(m): 
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)     
        nn.init.constant_(m.bias.data, 0)


# This module will train the generator and discriminator for a single epoch by traversing over the dataset
def train(gen, dis, dataloader, optim_G, optim_D, criterion, params, e):
    losses_g = []
    losses_d = []

    fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)

    real_label = 1.
    fake_label = 0.

    print("Starting training loop ")

    for i, data in enumerate(dataloader, 0):
        # if (i > 100):
        #     break
        # Updating the discriminator model          
        dis.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.shape[0]

        # Defining the label parameter for training 
        label = torch.full((b_size,), params['nz'], dtype=torch.float, device=device)
        label.fill_(real_label)
        # Forward pass of the discriminator model 
        label_pred = dis(real_cpu).view(-1)

        # Training the real images with the labels of the ones 
        loss_dx = criterion(label_pred, label)
        loss_dx.backward()

    
        # Training the discriminator with the fake images generated with the noise from generator 
        noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
        fake = gen(noise)
        label_pred = dis(fake.detach()).view(-1)
        label.fill_(fake_label)

        loss_dgz = criterion(label_pred, label)
        loss_dgz.backward()

        # Updating the D parameters 
        loss_d = loss_dx + loss_dgz
        optim_D.step()

        loss_d_val = loss_d.mean().item()
        losses_d.append(loss_d_val)


        # Training the generator model by updating the params with the generator loss 
        gen.zero_grad()
        label.fill_(real_label)
        label_pred = dis(fake).view(-1) # Again forward pass through the discriminator as the discriminator is updated once 

        loss_ggz = criterion(label_pred, label)     
        loss_ggz.backward()
        optim_G.step()      

        loss_g_val = loss_ggz.mean().item()
        losses_g.append(loss_g_val) 

        # Printing the logs only after certain iterations 
        if (i%100 == 0):
            output = "[{0}][{1}/{2}] Generator loss: {3:.3f}, Discriminator loss: {4:.3f}".format(e, i, len(dataloader), loss_g_val, loss_d_val)
            print(output)

    img_stack = []
    if (e % 1 == 0):
        with torch.no_grad():
            fake = gen(fixed_noise).detach().cpu()
            img_stack = vutils.make_grid(fake, padding=2, normalize=True)
            img_stack = img_stack.permute(1,2,0).numpy()*255
            img_stack = img_stack.astype(np.uint8)
            # print("img stack max: ", img_stack.max(), " img stack min: ", img_stack.min())
    
    return img_stack, losses_g, losses_d  
    
# MNIST DATASET LOADING
def get_mnist_train_val_loader(data_dir,
                    batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False,
                    augment=True):
    
    val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    

    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])

    train_dataset = datasets.MNIST( root=data_dir, train=True,download=True, transform=train_transform)
    val_dataset = datasets.MNIST( root=data_dir, train=True,download=True, transform=val_transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.8 * num_train))

    if shuffle:
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[:split], indices[split:]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory)
    val_loader = torch.utils.data.DataLoader( val_dataset, batch_size=batch_size, sampler=valid_sampler,
     num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader

def get_mnist_test_loader(data_dir,
                    batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False):
   
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    dataset = datasets.MNIST( root=data_dir, train=False,download=True, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory)

    return data_loader

# This is the main function to train the model for GAN on mnist dataset 
def train_model(params):
    gen = Generator(params['nz'], params['ndim'], params['nc']).to(device)
    dis = Discriminator(params['nc'], params['ndim']).to(device)

    gen.apply(weights_init)
    dis.apply(weights_init) 
    print("models initialized")

    mnist_train_loader, mnist_val_loader = get_mnist_train_val_loader('./train_data', batch_size=params['batch_size'], shuffle=True,
                    num_workers=params['num_workers'],pin_memory=False, augment=True )
                    
   

    #optim_G = optim.Adam(gen.parameters(), lr = params['lr_g'], betas = (0.5, 0.999))
    #optim_D = optim.Adam(dis.parameters(), lr = params['lr_d'], betas = (0.5, 0.999))

    optim_G = optim.SGD(gen.parameters(), lr = params['lr_g'], momentum=0.9)
    optim_D = optim.SGD(dis.parameters(), lr = params['lr_d'], momentum=0.9)

    criterion = nn.BCELoss()
    print("optimizers defined")  

    img_lists, gen_loss, disc_loss = [], [], []
    # Training loop
    for e in range(0, params['epochs']):
        img_stack, loss_g, loss_d = train(gen, dis, mnist_train_loader, optim_G, optim_D, criterion, params, e)     
        img_save_name = './figs/' + str(e) + '.png'
        print("img stack shape: ", img_stack.shape)
        cv2.imwrite(img_save_name, img_stack)
        print("Saving img at: ", img_save_name)
        gen_loss += loss_g
        disc_loss += loss_d

        ckpt_path = os.path.join(params['ckpt'], 'ckpt_' + str(e) + '.pth')
        model_dict = {}
        model_dict['gen'] = gen.state_dict()
        model_dict['dis'] = dis.state_dict()
        torch.save(model_dict, ckpt_path)

    # print("len list: ", len(losses_g))

    plt.figure(figsize=(10,5))
    plt.title("Generator vs Discriminator")
    plt.plot(gen_loss,label="Generator")
    plt.plot(disc_loss,label="Discriminator")   
    plt.xlabel("Iters")
    plt.ylabel("Loss")
    plt.legend()
    plt.show() 
    plt_save_name = './figs/train_process.png'
    plt.savefig(plt_save_name) 


        
def main():
    params = {}
    params['epochs'] = 10
    params['lr_g'] = 1e-3
    params['lr_d'] = 1e-3 
    params['batch_size'] = 64
    params['num_workers'] = 8
    params['nz'] = 64
    params['ndim'] = 64
    params['nc'] = 1
    params['ckpt'] = './ckpts'

    train_model(params) 

def generate_images(ckpt_path):
    model = torch.load(ckpt_path)
    gen = Generator(nz=64, ndim=64, nc=1)
    gen.load_state_dict(model['gen'])
    gen = gen.to(device)
    print("Model loaded : ", ckpt_path)

    # number of images to generatro
    ncols = 20
    nrows = 5
    N = nrows * ncols
    print('Generating {} images'.format(N))
    #Initializing a random latent
    z0 = torch.randn(nrows, 64, 1, 1, device=device)
    z1 = torch.randn(nrows, 64, 1, 1, device=device)
    # Create interpolations between two random latents z0 and z1
    
    for nid in range(0, nrows):
        # Now gz is contains 64 images 
        z = torch.zeros([ncols, 64, 1, 1], dtype=torch.float32).to(device)
        for id in range(0, ncols):
            t = (id/float(ncols-1)) 
            z[id] = z0[nid]*(1-t) + z1[nid]*t

        gz = gen(z).detach().cpu()

        # Now this Z array has interpolated values in between
        img_ = vutils.make_grid(gz, nrow=ncols, padding=2, normalize=True)
        img_ = img_.permute(1,2,0).cpu().numpy()*255   
        img_ = img_.astype(np.uint8)

        save_name = './figs/interpolations' + str(nid) + '.png'
        cv2.imwrite(save_name, img_)

if __name__ == "__main__":
    #main()
    generate_images('/data3/ankit/Coursework/DLCV/Assignment_2/ckpts_init/ckpt_9.pth') 