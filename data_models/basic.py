from abc import ABCMeta, abstractmethod
from statistics import stdev
import pandas as pd
import random
import numpy as np


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

    def __init__(self, data_path, split_percentages=[0.70, 0.20, 0.10]):
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

    def X_y_split(self, y):

        self.test_data_X = self.test_data.loc[:, self.test_data.columns != y]
        self.test_data_y = self.test_data.loc[:, y]
        self.calib_data_X = self.calib_data.loc[:, self.calib_data.columns != y]
        self.calib_data_y = self.calib_data.loc[:, y]
        self.train_data_X = self.train_data.loc[:, self.train_data.columns != y]
        self.train_data_y = self.train_data.loc[:, y]
        

    def get_stats(self):
        pass
    

class LOCOData(Base):
    """
    This class will allow us to replicate the experiment performed in the LOCO article, which we can hopefully 
    apply on other predicition sets
    """

    def __init__(self, n=1000, split_percentages=[0.70, 0.20, 0.10], interval=None, std=1):
        self.n = n
        self.split_percentages = split_percentages
        self.read_data(interval=interval, std=std)
        self.split()

    def mu_function(self, x):
        """ Additive model """

        f1 = np.sin(np.pi*(1+x[0]))*(x[0] < 0)
        f2 = np.sin(np.pi*x[1])
        f3 = np.sin(np.pi*(1+x[2]))*(x[2] > 0)

        return f1 + f2 + f3

    def read_data(self, interval=None, std=1):

        if not interval:
            x = np.random.uniform(-1, 1, size=(6, self.n)).T
        else:
            x1 = np.random.uniform(interval[0][0], interval[0][1], size=(1, self.n))
            x2 = np.random.uniform(interval[1][0], interval[1][1], size=(1, self.n))
            x3 = np.random.uniform(interval[2][0], interval[2][1], size=(1, self.n))
            x4 = np.random.uniform(interval[3][0], interval[3][1], size=(1, self.n))
            x5 = np.random.uniform(interval[4][0], interval[4][1], size=(1, self.n))
            x6 = np.random.uniform(interval[5][0], interval[5][1], size=(1, self.n))
            x = np.concatenate((x1, x2, x3, x4, x5, x6)).T
        err = np.random.normal(0, std, size=self.n)
        y = []
        for i, _ in enumerate(list(x)):
            y.append(self.mu_function(x[i])+err[i])

        self.data = pd.DataFrame(x, columns=['f1', 'f2', 'f3', 'f4', 'f5', 'f6'])
        self.data['Y'] = y

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

    def X_y_split(self, y):

        self.test_data_X = self.test_data.loc[:, self.test_data.columns != y]
        self.test_data_y = self.test_data.loc[:, y]
        self.calib_data_X = self.calib_data.loc[:, self.calib_data.columns != y]
        self.calib_data_y = self.calib_data.loc[:, y]
        self.train_data_X = self.train_data.loc[:, self.train_data.columns != y]
        self.train_data_y = self.train_data.loc[:, y]
        

    def get_stats(self):
        pass
    
    