import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

class bearing_dataset():
    """
    A dataset class that can be used for the Mikrocentrum Bearing casus. 
    Example:
    ds = bearing_dataset("dataset", "train_conditions.csv")
    >> print(ds.shape)
    
    ds_train, ds_test = ds.train_test_split()
    ds_train.shape, ds_test.shape
    >> ((1293,), (431,))
    
    print(ds_train[0].head())
    >>    
        0  0.032
        1  0.088
        2 -0.032
        3 -0.171
        4 -0.225
    """
    def __init__(self, path, labelfile, colnames = ["b4"]):
        self.path = path
        self.labelfile = labelfile
        
        # Get all the filenames in this path and remove all files without txt extensions
        self.files = os.listdir(path)
        self.files = pd.Series(self.files)
        self.files = list(self.files[self.files.str[-3:] == "txt"].values)
        
        self.ind = list(np.arange(len(self.files)))
        
        # Read the labels and assign colnames. These colnames can be changed by the user
        self.labels = pd.read_csv(os.path.join(path, labelfile), sep = ";")
        self.colnames = colnames
    
    def __str__(self):
        return "Class bearing_dataset with files from '{}' and size {}. It holds the following files: {} ... {}.\nGet full list of files with ds.files".format(self.path, self.shape, self.files[:3], self.files[-3:])
    
    def __getitem__(self, ind):
        """
        Return pandas dataframe
        """
        try:
            return pd.read_table(os.path.join(self.path, "{}.txt".format(ind)), names = self.colnames).b4
        except IndexError as e:
            raise IndexError("Index out of range with index '{}'".format(ind))
    
    def __len__(self):
        return len(self.files)
    
    def train_test_split(self, test_size = 0.25, random_state = None):
        # Train test split index wich will be used for splitting 
        ind = np.arange(len(self))
        train_ind, test_ind = train_test_split(ind, test_size = test_size, random_state = random_state)
        
        ds_train = bearing_dataset(self.path, self.labelfile, self.colnames)
        ds_test = bearing_dataset(self.path, self.labelfile, self.colnames)
        
        for subds, subind in [(ds_train, train_ind), (ds_test, test_ind)]:
            subds.files = list(np.array(self.files)[subind]) # doing a weird list to array to list conversion due to indexing supriority
            subds.labels = self.labels.loc[subind, :]
            subds.ind = list(subind)
        
        return ds_train, ds_test
    
    @property
    def shape(self):
        return (len(self),)