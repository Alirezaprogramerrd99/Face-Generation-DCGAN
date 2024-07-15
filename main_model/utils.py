import torch
from pathlib import Path
import pickle
from main_model.DCGAN_model import Generator, create_generator_discriminator
from main_model.constants import device, nz

import torchvision.transforms as transforms


def save_model(model: torch.nn.Module,
               target_dir: Path,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="mymodel.pth")
  """
  # Create target directory
  target_dir.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
  
  
def load_model(loaded_model: torch.nn.Module, target_dir: Path, model_name: str, GPU=False) -> torch.nn.Module:
  """loades a PyTorch model.

  Args:
    loaded_model: A target PyTorch model to load.
    target_dir: A directory for loading the model from.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.
    GPU: to whether save it on the GPU

  Example usage:
    save_model(loaded_model=loaded_model,
               target_dir="models",
               model_name="mymodel.pth")
  """
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  models_state_dict = torch.load(f=target_dir / model_name, map_location=device)
  loaded_model.load_state_dict(models_state_dict)
  loaded_model = loaded_model.to(device=device)

  return loaded_model

  
def write_list_to_file(filename, num_list):
  with open(filename, 'wb') as file:
    pickle.dump(num_list, file)

def read_list_from_file(filename):
    with open(filename, 'rb') as file:
        num_list = pickle.load(file)
    return num_list

def generate_image(generator: torch.nn.Module):
  fake_image = torch.randn(1, nz, 1, 1, device=device)

  with torch.no_grad():
    fake_image = generator(fake_image).detach().cpu()
  
  return fake_image

def tensor_to_pil_image(tensor: torch.Tensor):
    
    transform = transforms.Compose([ transforms.Resize(256), transforms.ToPILImage()])
    return transform(tensor.squeeze(0))  # Remove the batch dimension if present

# Save the PIL image as a JPG file
def save_image_as_jpg(image, filename):
    image.save(filename, 'JPEG')
