from abc import ABCMeta, abstractmethod

from scipy import rand
from conformal_prediction import core
import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import random


class BaseConformal(metaclass=ABCMeta):

    @abstractmethod
    def calibrate(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError()


class SimpleConformal(BaseConformal):

    def __init__(self, data_model, model, alpha, class_order):

        """
        :param class_order: Auxiliar dictionary used to specify the order in which we should associate classes and softmax output
        """
        
        self.data_model = data_model
        self.model = model
        self.alpha = alpha
        self.class_order = class_order

    def calibrate(self):
        """
        This is the base method used to estimate the lambda value based on the calibration set in data_model and alpha value
        """

        # Estimate the sofftmax probabilities of whatever model we want (it is important that the model class has a method fitted)
        pred_calib = self.model.predict_proba(self.data_model.calib_data_X)

        # First we will use the class order to assign the order which we expect from the softmax matrix
        calib_true_score = self.data_model.calib_data_y.to_numpy()
        mapped = np.vectorize(self.class_order.get)(calib_true_score)  

        # TODO: make nicer, there has to be a way to slice the matrix
        scores = []
        for i in range(len(mapped)):
            scores.append(pred_calib[i][mapped][0])

        # Sort scores for quantile estimation
        scores.sort()

        # Estimate the quantile based on the alpha value
        lambda_conformal = np.quantile(scores, 1-self.alpha, interpolation='lower')

        return lambda_conformal

    def predict(self, data, lambda_conformal):
        """
        Based on the calibrated lambda estimate the new conformal prediction sets
        """

        pred_data = self.model.predict_proba(data)

        pred = []
        inv_order_dict = {v: k for k, v in self.class_order.items()}
        for i in range(len(data)):
            temp = np.where(pred_data[i] > lambda_conformal)
            conformal_set = np.vectorize(inv_order_dict.get)(temp)[0]
            pred.append(conformal_set.tolist())

        return pred

    def evaluate(self, pred, true_data, plot=True):
        """
        This method will allow us to see whether the coverage conditions are fulfilled, the average size of the sets, etc.
        :param pred: output of the method predict
        :param true_data: self.data.test_data_y for example
        """

        assert len(pred) == len(true_data)

        coverage = sum([True if true_data.iloc[i] in pred[i] else False for i in range(len(true_data))])/len(pred)
        size = [len(pred[i]) for i in range(len(pred))]

        if plot:
            count = Counter(size)
            df = pd.DataFrame.from_dict(count, orient='index')
            df.plot(kind='bar', figsize=(12,8))
        
        size = sum(size)

        return coverage, size


class SplitConformal(BaseConformal):

    def __init__(self, alpha):

        self.alpha = alpha

    def calibrate(self, data_X, data_y, model, residual_type='normal', rand_state=42):
        """
        This is the base method used to estimate the lambda value based on the calibration set in data_model and alpha value
        """
        if rand_state:
            random.seed(rand_state)
            
        D1_index = random.sample(range(len(data_X)), int(len(data_X)*0.5))
        indexes = range(len(data_X))
        D2_index = list(set(indexes).difference(set(D1_index)))

        D1_X = data_X.iloc[D1_index]
        D2_X = data_X.iloc[D2_index]
        D1_y = data_y.iloc[D1_index]
        D2_y = data_y.iloc[D2_index]

        model.fit(D1_X, D1_y)       

        # Evaluate the residuals on the D2 using f1
        residuals = self.residual(y_true=D2_y, pred=model.predict(D2_X), type=residual_type).to_numpy()
        residuals.sort()

        n = len(data_y)

        k = int(np.ceil((n/2+1)*(1-self.alpha)))
        lambda_conformal = residuals[k]

        return lambda_conformal, model

    def residual(self, pred, y_true, type='normal'):
        
        if type=='normal':
            residuals = abs(y_true-pred)

        return residuals

    def predict(self, data, model, lambda_conformal):
        """
        Based on the calibrated lambda estimate the new conformal prediction sets
        """

        pred_data = model.predict(data)

        pred = []
        for i in range(len(data)):
            conformal_set = [pred_data[i]-lambda_conformal, pred_data[i]+lambda_conformal]
            pred.append(conformal_set)

        return pred

    def evaluate(self, pred, true_data, plot=True):
        """
        This method will allow us to see whether the coverage conditions are fulfilled, the average size of the sets, etc.
        :param pred: output of the method predict
        :param true_data: self.data.test_data_y for example
        """

        assert len(pred) == len(true_data)

        coverage = sum([True if (true_data.iloc[i] <= pred[i][1]) & (true_data.iloc[i] >= pred[i][0]) else False for i in range(len(true_data))])/len(pred)
        size = [pred[i][1]-pred[i][0] for i in range(len(pred))]

        # Implement a more representative plot
        if plot:
            pass
        
        size = sum(size)

        return coverage, size
