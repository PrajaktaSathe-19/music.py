import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

!unzip "/content/gan dataset.zip"

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)
    
# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


z_dim = 5 #random noise dimension
image_dim = 128 * 128 * 3
batch_size = 33
num_epochs = 30

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)

train_transforms = transforms.Compose(
    [
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.42,0.37,0.31), (0.24,0.21,0.19)),
    ]
)


dataset = ImageFolder(root= "gan dataset", transform=train_transforms)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#Optimizer: Adam -> Adaptive Moment Estimation
optimizer_disc = torch.optim.Adam(disc.parameters(), lr=3e-4)
optimizer_gen  = torch.optim.Adam(gen.parameters(), lr=3e-4)

loss_fn = nn.BCELoss()

for epoch in range(num_epochs):   #Number of epochs

    #Loop using Batches of real images (real) and their corresponding labels (_) over dataloader:
    for batch_index, (real, _) in enumerate(loader):
        real = real.view(batch_size, image_dim).to(device) #Batch of RGB images Flattened to 1D Tensor



        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))  |  x -> real ; z -> noise

        noise = torch.randn(batch_size, z_dim).to(device) #defining random noise vector
        fake = gen(noise) #generate fake images from noise

        disc_real = disc(real).view(-1) #score of real images through Discriminator
        lossD_real = loss_fn(disc_real, torch.ones_like(disc_real)) #Maximising BCELoss of Discrimator score(Real) & Target Label 1

        disc_fake = disc(fake).view(-1) #score of fake images through Discriminator
        lossD_fake = loss_fn(disc_fake, torch.zeros_like(disc_fake)) #Maximising BCELoss of Discrimator score(Fake) & Target Label 0

        lossD = (lossD_real + lossD_fake) / 2

        disc.zero_grad()                           #resets the gradients of the discriminator's parameters to zero.
        lossD.backward(retain_graph=True)          #We are reusing fake =gen(noise)
        optimizer_disc.step()



        # Train Generator: min log(1 - D(G(z)))
        # i.e,  max log(D(G(z))

        d_fake = disc(fake).view(-1)  #score of fake images through Discriminator
        lossG = loss_fn(d_fake, torch.ones_like(d_fake))  ## Maximising BCELoss of Discrimator score(Fake) & Target Label 1, as Generator wants Discriminator to misclassify

        gen.zero_grad()
        lossG.backward()
        optimizer_gen.step()



        if batch_index == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_index+1}/{len(loader)} \
                      Loss Discriminator: {lossD:.4f}, loss Generator: {lossG:.4f}"
            )




            #Visualize images
            with torch.no_grad():                                 #Pause Gradient_descent during block

                fixed_noise = torch.randn((batch_size, z_dim)).to(device) ## Generate a fixed noise vector for visualization

                fake = gen(fixed_noise).reshape(-1, 3, 128, 128)
                data = real.reshape(-1, 3, 128, 128)

                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                # Convert the image tensors to numpy arrays
                img_fake = img_grid_fake.permute(1, 2, 0).cpu().numpy()
                img_real = img_grid_real.permute(1, 2, 0).cpu().numpy()

                # Plot the real and fake images side by side
                fig, axes = plt.subplots(1, 2, figsize=(20, 10))
                axes[0].imshow(img_real)
                axes[0].set_title("Real Images")
                axes[0].axis('off')

                axes[1].imshow(img_fake)
                axes[1].set_title("Fake Images")
                axes[1].axis('off')

                plt.show()

