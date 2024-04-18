import argparse
import os
from pathlib import Path
import pickle
import sys

from datasets import Dataset, DatasetDict, load_dataset
import numpy as np
from google.colab import drive
drive.mount('/content/drive')


wikitext_dataset = load_dataset("wikipedia", language="es", date="20240401")

dataset_path = '/content/drive/My Drive/Tesis/wikitext_dataset'

# Ensure the directory exists
Path(dataset_path).mkdir(parents=True, exist_ok=True)

# Save the dataset to disk
wikitext_dataset.save_to_disk(dataset_path)

print(f'Dataset saved to {dataset_path}')