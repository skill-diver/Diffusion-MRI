import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
from scipy import ndimage

class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=False):
        super().__init__()
        self.directory = directory

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)

        self.filelist = sorted(os.listdir(directory))

    def __getitem__(self, x):
        # load just the slice we need
        # filepath = self.filepaths[x]
        # idx = self.indices[x]
        # loaded_data = np.load(filepath, mmap_mode='r')
        # image = torch.tensor(loaded_data['images'][idx]).float()
        # label = torch.tensor(loaded_data['labels'][idx])[:, None, ...].float()
        # weak_label = torch.tensor([loaded_data['weak_labels'][idx]]).float()
        # out_dict = {"y": weak_label}
        # number = 0
        filepath = os.path.join(self.directory, self.filelist[x])
        print("Loading file:", self.filelist[x])
        loaded_data = np.load(filepath)
        image = torch.tensor(loaded_data['image']).float()
        label = torch.tensor(loaded_data['label'])[None, ...].float()
        weak_label = loaded_data['weak_label']
        out_dict = {"y": weak_label}
        number = int(self.filelist[x].split('_')[2].split('.')[0])

        if self.test_flag:
            return (image, out_dict, weak_label, label, number)
        else:
            return (image, out_dict, weak_label)

    def __len__(self):
        return len(self.filelist)