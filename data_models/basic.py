from abc import ABCMeta, abstractmethod
from statistics import stdev
import pandas as pd
import random
import numpy as np
from sklearn.datasets import make_classification


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


class LinearData(Base):
    """
    This class will allow us to generate data which can be modelled using a linear model
    """

    def __init__(self, n=1000, split_percentages=[0.70, 0.20, 0.10], std=1, weights=None, num_features=None):
        self.n = n
        self.split_percentages = split_percentages
        self.read_data(std=std, weights=weights, num_features=num_features)
        self.split()

    def linear_function(self, x, weights=None):
        """ Additive model """

        if not isinstance(weights, np.ndarray):
            weights = np.random.uniform(low=-10, high=10, size=len(x))
        out = np.sum(x*weights)

        return out

    def read_data(self, std=1, weights=None, num_features=None):

        if isinstance(weights, np.ndarray):
            num_features = len(weights)
    
        x = np.random.uniform(-1, 1, size=(num_features, self.n)).T

        err = np.random.normal(0, std, size=self.n)
        y = []
        for i, _ in enumerate(list(x)):
            y.append(self.linear_function(x[i], weights=weights)+err[i])

        columns = ['f'+str(i+1) for i in range(num_features)]
        self.data = pd.DataFrame(x, columns=columns)
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


class LinearDataCorrelated(Base):
    """
    This class will allow us to generate data which can be modelled using a linear model
    """

    def __init__(self, n=1000, split_percentages=[0.70, 0.20, 0.10], mu=None, std=None, cov_matrix=None, weights=None, num_features=None):
        self.n = n
        self.split_percentages = split_percentages
        self.read_data(mu=mu, cov_matrix=cov_matrix, weights=weights, num_features=num_features, std=std)
        self.split()

    def linear_function(self, x, weights=None):
        """ Additive model """

        if not isinstance(weights, np.ndarray):
            weights = np.random.uniform(low=-10, high=10, size=len(x))
        out = np.sum(x*weights)

        return out

    def read_data(self, mu, cov_matrix, weights, std=1, num_features=None):

        if isinstance(weights, np.ndarray):
            num_features = len(weights)
    
        rng = np.random.default_rng()
        x = rng.multivariate_normal(mu, cov_matrix, size=self.n)

        err = np.random.normal(0, std, size=self.n)
        y = []
        for i, _ in enumerate(list(x)):
            y.append(self.linear_function(x[i], weights=weights)+err[i])

        columns = ['f'+str(i+1) for i in range(num_features)]
        self.data = pd.DataFrame(x, columns=columns)
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
    
    

class MultiClass(Base):

    def __init__(self, n=10000, n_features=10, n_informative=8, n_classes=10, split_percentages=[0.70, 0.20, 0.10], class_sep=1):
        """
        :param data_path: Path in which we can find the tabular data
        :param split: List with the percentage of splits following the convention [train, calibration, test]
        """
        self.split_percentages = split_percentages
        self.data = None
        self.read_data(n, n_features, n_informative, n_classes, class_sep)
        self.split()

    def read_data(self, n, n_features, n_informative, n_classes, class_sep):
        
        X, y = make_classification(n_samples=n, n_features=n_informative, n_informative=n_informative, n_redundant=0, n_repeated=0, n_classes=n_classes, class_sep=class_sep)
        uninformative = n_features-n_informative
        X_uninformative = np.random.normal(size=(uninformative, n)).T
        X = np.concatenate((X, X_uninformative), axis=1)
        columns = ['f'+str(i+1) for i in range(X.shape[1])]
        self.data = pd.DataFrame(X, columns=columns)
        self.data['y'] = y

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