from abc import ABCMeta, abstractmethod
import pandas as pd
import random


class Base(metaclass=ABCMeta):

    @abstractmethod
    def read_data(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def split(self, *args, **kwargs):
        raise NotImplementedError()

    def get_stats(self, *args, **kwargs):
        raise NotImplementedError()


class Tabular(Base):

    def __init__(self, data_path, split_percentages=[70, 20, 10]):
        """
        :param data_path: Path in which we can find the tabular data
        :param split: List with the percentage of splits following the convention [train, calibration, test]
        """
        self.data_path = data_path
        self.split_percentages = split_percentages
        self.read_data()
        self.split()

    def read_data(self):
        
        self.data = pd.read_csv(self.data_path)

    def split(self):
        """
        Method to split data into train, calibration and test.
        Train will be used to train the model.
        Calibration will be used for the conformal prediction estimation
        Test will be used to test our results
        """

        train_calib = random.sample(range(len(self.data)), 
                                    int(len(self.data)*(self.split_percentages[0]+self.split_percentages[1])))
        calib = random.sample(train_calib, 
                              int(len(self.data)*self.split_percentages[1]))

        indexes = range(len(self.data))
        test_indexes = set(indexes).difference(set(train_calib))
        calib_indexes = calib
        train_indexes = set(train_calib).difference(set(calib))

        self.test_data = self.data.loc[test_indexes]
        self.calib_data = self.data.loc[calib_indexes]
        self.train_data = self.data.loc[train_indexes]

    def X_y_split(self, y='quality'):

        self.test_data_X = self.test_data.loc[:, self.test_data.columns != y]
        self.test_data_y = self.test_data.loc[:, y]
        self.calib_data_X = self.calib_data.loc[:, self.calib_data.columns != y]
        self.calib_data_y = self.calib_data.loc[:, y]
        self.train_data_X = self.train_data.loc[:, self.train_data.columns != y]
        self.train_data_y = self.train_data.loc[:, y]
        

    def get_stats(self):
        pass
    

