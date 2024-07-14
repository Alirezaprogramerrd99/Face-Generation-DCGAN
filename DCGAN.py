import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from torchvision.datasets.utils import *
from pathlib import Path

from main_model.data_prepare import *
from main_model.DCGAN_model import *
from main_model.data_setup import create_dataloader
from main_model.constants import *
from main_model.utils import *
from main_model.engine import train_step
import logging


# Set random seed for reproducibility
# # Set random seeds
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
logging.basicConfig(filename='./DCGAN_log.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    manualSeed = 48
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True) # Needed for reproducible results

    data_path = Path("celeba_dataset")
    image_path = data_path / "celeba"

    prepare_data(data_path, image_path, "img_align_celeba")
    walk_through_dir(image_path / "img_align_celeba")

    transform = transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

    dataloader = create_dataloader(train_dir=image_path, transform=transform, batch_size= batch_size)
    # batch = next(iter(dataloader))
    # print("first batch's shape:", batch[0].shape)

    netG = create_generator_discriminator(ngpu, device, True, True)
    netD = create_generator_discriminator(ngpu, device, False, True)

    print("\nThe Generator's architecture: ",netG, True)
    print("\nThe Discriminator's architecture: ", netD, True)

        # Initialize the ``BCELoss`` function
    loss = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator (For keeping track of the generator’s learning progression, we will generate a fixed batch of latent vectors that are drawn from a Gaussian distribution (i.e. fixed_noise))
    # In the training loop, periodically this will be inputed fixed_noise into  G, and over the iterations we will see images form out of the noise.
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    print("fixed_noise's shape: ", fixed_noise.shape)

    print("Starting Training Loop...")
    logging.info("Starting Training Loop...")

    img_list = []
    G_losses = []
    D_losses = []

    for epoch in tqdm(range(num_epochs)):
        G_loss, D_loss = train_step(netD=netD, netG=netG, 
                                    epoch=epoch, dataloader=dataloader,
                                    loss=loss, optimizerD=optimizerD,
                                    optimizerG=optimizerG, img_list=img_list,
                                    fixed_noise=fixed_noise, real_label=real_label,
                                    fake_label=fake_label, device=device)
        G_losses.extend(G_loss)
        D_losses.extend(D_loss)
    
    models_info_path = Path("models")
    save_model(netD, models_info_path, "Generator.pth")
    save_model(netG, models_info_path, "Discriminator.pth")

    write_list_to_file(models_info_path / "G_losses.pkl", G_losses)
    write_list_to_file(models_info_path / "D_losses.pkl", D_losses)

    print("Models' trainning has been finished and saved.")
    logging.info("Models' trainning has been finished and saved.")
    