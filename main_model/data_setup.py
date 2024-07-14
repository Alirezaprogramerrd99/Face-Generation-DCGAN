from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from constants import NUM_WORKERS

def create_dataloader(train_dir: str, 
                      transform: transforms.Compose, 
                      batch_size: int, 
                      num_workers: int=NUM_WORKERS):
    
    dataset = dset.ImageFolder(root=train_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return dataloader
