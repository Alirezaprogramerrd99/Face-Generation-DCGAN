import zipfile
from pathlib import Path
import os

def prepare_data(data_path: Path, image_path: Path, zipfilename: str):
    print("The data path: ", data_path)
    print("The image path: ", image_path)

    if not image_path.exists():
        image_path.mkdir(parents=True, exist_ok=True)
        print(f"Did not find {image_path} directory, creating one ...")

    if not (image_path / zipfilename).exists():

        with zipfile.ZipFile(data_path / (zipfilename + ".zip"), "r") as zip_ref:
            print("unzipping the dataset ...")
            zip_ref.extractall(path=image_path) # unzipping from google drive to the image_path in colab's storage...
            print("Done!")

def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.
  Args:
    dir_path (str or pathlib.Path): target directory

  Returns:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  print (f"In ( {dir_path} ):")
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(filenames)} images in '{dirpath}'.")
