from torch.utils.data import Dataset
import numpy as np



class MouseDataset(Dataset):
    """
     Dataset for Caltech Mouse Social Interactions (CalMS21) Dataset.
     download from :  https://data.caltech.edu/records/1991
    """

    def __init__(self, data_path, ann_path=None):
        self.ann = np.load(ann_path, allow_pickle=True).item()
        self.data = np.load(data_path, allow_pickle=True).item()

    def __len__(self):
        return len(self.data.keys())



    #TODO return mask here
    def __getitem__(self, idx):
        seq_key = self.data.keys()[idx]
        if self.ann_path is not None:
            return self.data.values()[idx], seq_key, self.ann['sequences'][seq_key]['anntations']
        else:
            return self.data.values()[idx], seq_key
