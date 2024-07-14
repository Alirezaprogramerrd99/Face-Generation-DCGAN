import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import torchvision.utils as vutils
from main_model.constants import num_epochs, nz
import logging

logging.basicConfig(filename='./DCGAN_log.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def train_step(netD: torch.nn.Module, 
               netG: torch.nn.Module,
               epoch: int,
               dataloader: torch.utils.data.DataLoader, 
               loss: torch.nn.Module, 
               optimizerD: torch.optim.Optimizer,
               optimizerG: torch.optim.Optimizer,
               img_list: List,
               fixed_noise: torch.Tensor,
               real_label: float,
               fake_label: float,
               device: torch.device) -> Tuple[List, List]:
    
    D_losses, G_losses = [], []
    iters = 0
    
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_images = data[0].to(device)
        batch_size = real_images.size(0)
        label = torch.ones((batch_size, ), dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_images).view(-1)
        # Calculate loss on all-real batch
        errD_real = loss(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()
        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake_images = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake_images.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = loss(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake_images).view(-1)
        # Calculate G's loss based on this output
        errG = loss(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            logging.info('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        iters += 1

    return G_losses, D_losses