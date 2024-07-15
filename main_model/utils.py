import torch
from pathlib import Path
import pickle

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
  
def write_list_to_file(filename, num_list):
  with open(filename, 'wb') as file:
    pickle.dump(num_list, file)

def read_list_from_file(filename):
    with open(filename, 'rb') as file:
        num_list = pickle.load(file)
    return num_list