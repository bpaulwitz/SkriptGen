import cv2
import os
from numpy.lib.arraysetops import isin
import torch
import numpy as np
import pandas as pd

import sentence_tokens as st

from torch.utils.data import Dataset
from torchvision import transforms
from typing import Union

def save_dataset_render(render: Union[torch.Tensor, np.array], path: str) -> None:
    """
    Saves an render element from a Dataset_SkriptGen sample to disk
    @param render: render from sample
    @param path: path where to save the image
    """
    # make sure there is only one image in the batch
    assert(render.shape[0] == 1)

    # remove batch dimension
    render = render[0, :, :, :]

    if isinstance(render, torch.Tensor):
        # convert to numpy
        render = render.numpy()
        render = render.transpose((1, 2, 0)).astype(np.uint8)

    cv2.imwrite(path, render)


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
        render = render.transpose((2, 0, 1)).astype(np.float32)

        return {'render': torch.from_numpy(render), 'target_corpus': torch.from_numpy(target_corpus).long(), 'target_numbers': torch.from_numpy(target_numbers).float()}

class Dataset_ScriptGen(Dataset):
    def __init__(self, csv_file: str, data_root: str, encoding = None, max_len_encoding: int = None, max_len_floats: int = None, transform: transforms.Compose = None, file_ending = '.py') -> None:
        """
        Constructor
        @param csv_file: path to the csv-file. It has to have the file names of the renders in it's first column and the file names of the scripts in it's second column
        @param data_root: root directory for the dataset data. It has to contain a subdirectory 'Renders' with the rendered images of the data objects and a subdirectory 'Scripts' with the target scripts that generate the rendered object in blender
        @param encoding: encoding dictionary. Can be None (is created from the 'Scripts' subdirectory in data_root), a string (reads string as path to saved encoding script and loads it) or a dict (then the dictionary is the encoding)
        @param max_len_encoding: maximum length of the encoded script
        @param max_len_floats: maximum length of the floating point list for the encoded script
        @param transform: transformations of the sample. Is a torchvision.transforms.Compose object
        @param file_ending: filename ending of the target file (e.g. '.py' when the training targets are python scripts)
        """
        super().__init__()

        self.csv_file = csv_file
        self.csv = pd.read_csv(csv_file)
        self.render_dir = os.path.join(data_root, 'Renders')
        self.script_dir = os.path.join(data_root, 'Scripts')
        self.transform = transform

        if max_len_encoding is not None and max_len_floats is not None:
            self.max_len_encoding = max_len_encoding
            self.max_len_floats = max_len_floats
        else:
            encoding = None

        # make sure that these folders exist
        if not os.path.isdir(self.script_dir):
            raise FileNotFoundError("Could not find " + self.script_dir)
        if not os.path.isdir(self.render_dir):
            raise FileNotFoundError("Could not find " + self.render_dir)

        # if encoding is given by a file path (str)
        if isinstance(encoding, str):
            self.encoding, self.max_len_encoding, self.max_len_floats = st.load_sentence_encoding(encoding)

        # if we need to create an encoding (or need to compute the maximum length of the encoding/ float list)
        elif encoding is None or max_len_encoding is None and max_len_floats is None:
            all_scripts = []

            # collect all filepaths of scripts
            for file in os.listdir(self.script_dir):
                if file.endswith(file_ending):
                    all_scripts.append(os.path.join(self.script_dir, file))
            
            # create encoding
            self.encoding, self.max_len_encoding, self.max_len_floats = st.create_sentence_encoding(all_scripts)

        elif isinstance(encoding, dict):
            self.encoding = encoding

        # if encoding is neither None, a string or a dict
        else:
            raise ValueError("ERROR: encoding is not a string (load encoding file), a dictionary (use as encoding) or None (create encoding from scratch)!")

    def __len__(self) -> int:
        """
        get the amount of samples in this dataset
        @return: amount of samples in this dataset
        """
        return len(self.csv)

    def render_check(self):
        """
        Sometimes renders are white for some reason and fortunately they can't be resized by opencv then. Use this method to check for broken renders
        """
        faulty_indeces = []
        for i in range(len(self.csv)):
            render_path = os.path.join(self.render_dir, self.csv.iloc[i, 0])
            render = cv2.imread(render_path, cv2.IMREAD_UNCHANGED)
            try:
                cv2.resize(render, (256, 256))
                continue
            except Exception as e:
                print(e)
                faulty_indeces.append(i)

            #print(faulty_indeces)
            #exit()
        return faulty_indeces

    def remove_faulty_indices(self, indices: list = None):
        if indices is None:
            indices = self.render_check()

        # sort indices from highest to lowest
        indices = sorted(indices, reverse=True)
        
        with open(self.csv_file, 'r+') as csv:
            lines = csv.readlines()
            csv.seek(0)
            for i in indices:
                # add 1 becouse of the first line (which does not include data)
                lines.pop(i + 1)
            csv.writelines(lines)
            csv.truncate()


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
        encoded_script, target_numbers = st.encode_sentence_script(
            file_path=script_path, 
            encoding=self.encoding, 
            batch_length_skript=self.max_len_encoding,
            batch_length_numbers=self.max_len_floats)

        # create sample
        sample = {'render': render, 'target_corpus': np.array(encoded_script, dtype="int32"), 'target_numbers': np.array(target_numbers, dtype="float")}

        # add transform if necessary
        if self.transform:
            sample = self.transform(sample)

        return sample
