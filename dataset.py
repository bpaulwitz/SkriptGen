import cv2
import os
import torch
import numpy as np
import pandas as pd

import sentence_tokens as st

from torch.utils.data import Dataset
from torchvision import transforms

class Rescale(object):
    """
    Transformation to rescale the image of a sample from Dataset_ScriptGen
    """
    def __init__(self, height: int, width: int):
        """
        Constructor
        @param height: height of the rescaled image
        @param width: width of the rescaled image
        """
        self.height = height
        self.width = width

    def __call__(self, sample: dict) -> dict:
        render = sample['render']
        new_render = cv2.resize(render, (self.height, self.width))

        return {'render': new_render, 'target_corpus': sample['target_corpus'], 'target_numbers': sample['target_numbers']}

class ToTensor(object):
    """
    Convert numpy values of a sample from Dataset_ScriptGen to pytorch tensors
    """
    def __call__(self, sample: dict) -> dict:
        render, target_corpus, target_numbers = sample['render'], sample['target_corpus'], sample['target_numbers']
        # numpy image: H x W x C
        # torch image: C X H X W
        render = render.transpose((2, 0, 1))

        return {'render': torch.from_numpy(render), 'target_corpus': torch.from_numpy(target_corpus), 'target_numbers': torch.from_numpy(target_numbers)}

class Dataset_ScriptGen(Dataset):
    def __init__(self, csv_file: str, data_root: str, encoding: dict = None, transform: transforms.Compose = None) -> None:
        """
        Constructor
        @param csv_file: path to the csv-file. It has to have the file names of the renders in it's first column and the file names of the scripts in it's second column
        @param data_root: root directory for the dataset data. It has to contain a subdirectory 'Renders' with the rendered images of the data objects and a subdirectory 'Scripts' with the target scripts that generate the rendered object in blender
        @param encoding: encoding dictionary. Can be None (is created from the 'Scripts' subdirectory in data_root), a string (reads string as path to saved encoding script and loads it) or a dict (then the dictionary is the encoding)
        @param transform: transformations of the sample. Is a torchvision.transforms.Compose object
        """
        super().__init__()

        self.csv = pd.read_csv(csv_file)
        self.render_dir = os.path.join(data_root, 'Renders')
        self.script_dir = os.path.join(data_root, 'Scripts')
        self.transform = transform

        # make sure that these folders exist
        if not os.path.isdir(self.script_dir):
            raise FileNotFoundError("Could not find " + self.script_dir)
        if not os.path.isdir(self.render_dir):
            raise FileNotFoundError("Could not find " + self.render_dir)

        # if we need to create an encoding
        if encoding is None:
            all_scripts = []

            # collect all filepaths of scripts
            for file in os.listdir(self.script_dir):
                if file.endswith('.py'):
                    all_scripts.append(os.path.join(self.script_dir, file))
            
            # create encoding
            self.encoding = st.create_sentence_encoding(all_scripts)

        # if encoding is given by a file path (str)
        elif type(encoding) is str:
            self.encoding = st.load_sentence_encoding(encoding)

        # if encoding is neither None, a string or a dict
        elif type(encoding) is not dict:
            raise ValueError("ERROR: encoding is not a string (load encoding file), a dictionary (use as encoding) or None (create encoding from scratch)!")

    def __len__(self) -> int:
        """
        get the amount of samples in this dataset
        @return: amount of samples in this dataset
        """
        return len(self.csv)

    def __getitem__(self, index: int) -> dict:
        """
        get the sample from the csv-file at index
        @param index: index in csv-file
        @return: sample dictionary with {'render': render_image, 'target_corpus': target_script_blanked_numbers, 'target_numbers': target_numbers_for_target_corpus}
        """
        if torch.is_tensor(index):
            index = index.tolist()

        # path to render image -> the first item in the csv file
        render_path = os.path.join(self.render_dir, self.csv.iloc[index, 0])
        render = cv2.imread(render_path, cv2.IMREAD_UNCHANGED)

        # path to script -> second item in the csv file
        script_path = os.path.join(self.script_dir, self.csv.iloc[index, 1])
        encoded_script, target_numbers = st.encode_sentence_script(script_path, self.encoding)

        # create sample
        sample = {'render': render, 'target_corpus': np.array(encoded_script, dtype="int32"), 'target_numbers': np.array(target_numbers, dtype="float")}

        # add transform if necessary
        if self.transform:
            sample = self.transform(sample)

        return sample
