from calendar import different_locale
import copy
import random
import numpy as np
import scipy
import matplotlib.pyplot as plt
from conformal_prediction.basic import SplitConformal

""" 
We will implement 3 alternatives for LOCO:

1. Estimate the feature importance by generating multiple samples and estimaing the mean importance/median (similar to what we did in SL)
2. Replicate the first approach mentioned in the LOCO paper, i.e. estimate the prediction with and without and consider the estimated conformal prediction set to get the confidence interval
3. Our method, which also relies on the conformal prediction but can be applied to discrete and continuous variables

"""

class SimpleLOCO():

    def __init__(self, model, train_data, y_feature):
        
        self.model = model
        self.train_data = train_data
        self.y_feature = y_feature

    def run_loco(self, variable, prop=0.7, loss_type='normal', alpha=0.05, bootstrap=1000):
        """
        This is the method which wraps all the steps required. The steps are the following:
        
        1. Split the data in two complementary sets (D1 and D2) 
        2. Run the model on the training data D1 (obtaining f1)
        3. Remove the feature "variable" from the dataset D1 and rerun the same model (obtaining f1-j)
        4. Predict results on D2 by using f1
        5. Predict results on D2 removing the feature "variable" using the model f1-j
        6. Calculate the feature importance value which is abs(D2_y-f1-j_y)-abs(D2_y-f1_y) (coming from point point 5 and 4 respectively)
        7. We can then estimate the confidence interval around this quantity using a non-parametric bootstrap approach.
        
        :param variable: The variable over which we want to estimate the importance
        """

        D1_X, D1_y, D2_X, D2_y = self._split(self.train_data, prop=prop)
        f1 = self.model.fit(D1_X, D1_y)
        pred_D2 = self.model.predict(f1, D2_X, softmax=False)
        f1_j = self.model.fit(D1_X.loc[:, D1_X.columns != variable], D1_y)
        pred_D2_j = self.model.predict(f1_j, D2_X.loc[:, D2_X.columns != variable], softmax=False)

        feature_importance = self.loss(true_y=D2_y, 
                                       pred_y=pred_D2, pred_y_j=pred_D2_j, type=loss_type)

        # Next thing is to estimate the confidence interval using bootstrap
        estimate = []
        for _ in range(bootstrap):

            idx = random.choices(range(len(D2_X)), k=len(D2_X))
            D2_X_boot = D2_X.iloc[idx, D2_X.columns != variable]
            D2_y_boot = D2_y.iloc[idx]

            pred_D2_j_boot = self.model.predict(f1_j, D2_X_boot, softmax=False)
            pred_D2_boot = pred_D2[idx]

            estimate.append(self.loss(true_y=D2_y_boot, 
                                      pred_y=pred_D2_boot, pred_y_j=pred_D2_j_boot, 
                                      type=loss_type))

        ci = self._confidence_interval(boot_estimate=estimate, true_estimate=feature_importance, alpha=alpha)

        return feature_importance, ci


    def _confidence_interval(self, boot_estimate, true_estimate, alpha):
        """
        Compute the confidence interval based on the bootstrap non-parametric
        """
        se = np.std(boot_estimate)
        ci_lower = true_estimate - scipy.stats.norm.ppf(1-alpha/2)*se
        ci_upper = true_estimate + scipy.stats.norm.ppf(1-alpha/2)*se
        
        return [ci_upper, ci_lower] 
                
    def loss(self, true_y, pred_y, pred_y_j, type):
        """
        This method will compute different loss functions for the estimation of the feature importance variable. It will allow us 
        to implement multiple different loss functions depending on whether the variables are continuous or discrete
        """

        if type ==  'normal':
            loss_value = np.median(abs(true_y-pred_y_j)-abs(true_y-pred_y))
        elif type == 'mean':
            loss_value = np.mean(abs(true_y-pred_y_j)-abs(true_y-pred_y))

        return loss_value

    def _split(self, data, prop=0.7):
        """
        :param data: Data needs to be in pandas format
        :param prop: Percentage of data used for the split
        """

        D1_index = random.sample(range(len(data)), int(len(data)*prop))

        indexes = range(len(data))
        D2_index = list(set(indexes).difference(set(D1_index)))

        D1 = data.iloc[D1_index]
        D2 = data.iloc[D2_index]

        D1_X = D1.loc[:, D1.columns != self.y_feature]
        D1_y = D1.loc[:, self.y_feature]
        D2_X = D2.loc[:, D1.columns != self.y_feature]
        D2_y = D2.loc[:, self.y_feature]

        return D1_X, D1_y, D2_X, D2_y


class OldConformalLOCO():

    def __init__(self, data_model, conformal_model):
        
        self.data_model = data_model
        self.conformal_model = conformal_model

    def run(self, model, variable, lambda_conformal, loss_type='normal', plot=True):
        """
        This is the method that runs the variable importance method form the LOCO paper. The steps followed are (which might be wrong):
        1. Fit model f and f_j on train data such that f is fitted over all the dataset, and f_j does not take into consideration one variable j
        2. Predict mu and mu_j on all the instances of the test data, mu on all the features and mu_j on all except feature j
        3. Predict the conformal sets on the test data X using the lambda_conformal estimated before for the complete training data
        4. Calculate W_j
        
        The main issue is that the confidence intervals are symetric        
        """

        # Prediction of the complete model
        model_all = copy.copy(model)
        model_j = copy.copy(model)

        model_all.fit(self.data_model.train_data_X, self.data_model.train_data_y)
        model_j.fit(self.data_model.train_data_X.loc[:, self.data_model.train_data_X.columns != variable], 
                                  self.data_model.train_data_y)

        pred_all = model_all.predict(self.data_model.test_data_X)
        pred_j = model_j.predict(self.data_model.test_data_X.loc[:, self.data_model.test_data_X.columns != variable])

        conformal_set = self.conformal_model.predict(data=self.data_model.test_data_X, lambda_conformal=lambda_conformal)
        
        W_j = self._loss(conformal_set=np.array(conformal_set), pred_all=pred_all, pred_j=pred_j, type=loss_type)

        if plot:
            import matplotlib.pyplot as plt

            x = self.data_model.test_data_X.loc[:, variable]
            y = W_j

            plt.figure(figsize=(12, 8))
            plt.plot(x, [i for (i,j) in y], 'rs', markersize = 4)
            plt.plot(x, [j for (i,j) in y], 'bo', markersize = 4)
            #plt.plot((x,x),([i for (i,j) in y], [j for (i,j) in y]),c='black')

            plt.xlabel('Location')
            plt.ylabel('W')
            plt.title('Variable Importance for feature \n ' + variable)
            plt.show()

        return W_j


    def _loss(self, conformal_set, pred_all, pred_j, type='normal'):

        if type == 'normal':
            W_j = abs(conformal_set - pred_j.reshape((len(pred_j), 1))) - abs(conformal_set - pred_all.reshape((len(pred_all), 1)))

        return W_j
    


class NewConformalLOCO():

    def __init__(self, conformal_predictor):
        
        self.conformal_predictor = conformal_predictor

    def run(self, model, data_X_calibrate, data_y_calibrate, data_X_test,
            variable, loss_type='basic', plot=10, test_same_dist=True):
        """
        
        """

        # TODO: How can we improve this?
        data_X_j = copy.copy(data_X_calibrate)
        data_X_j[variable] = 0
        data_X_test_j = copy.copy(data_X_test)
        if test_same_dist:
            data_X_test_j[variable] = 0

        lambda_conformal_all, model_all = self.conformal_predictor.calibrate(data_X=data_X_calibrate, data_y=data_y_calibrate, 
                                                                  model=model, residual_type='normal')
        lambda_conformal_j, model_j = self.conformal_predictor.calibrate(data_X=data_X_j, 
                                                                data_y=data_y_calibrate, 
                                                                model=model, residual_type='normal')

        conformal_sets_all = self.conformal_predictor.predict(data=data_X_test, model=model_all, lambda_conformal=lambda_conformal_all)
        conformal_sets_j = self.conformal_predictor.predict(data=data_X_test_j, model=model_j, lambda_conformal=lambda_conformal_j)

        if plot:
            y_j = conformal_sets_j[:plot]
            y_all = conformal_sets_all[:plot]
            x = range(len(y_j))

            plt.figure(figsize=(12, 8))
            plt.plot(x, [i for (i,j) in y_j], 'rs', markersize = 4)
            plt.plot(x, [j for (i,j) in y_j], 'rs', markersize = 4)
            plt.plot(x, [i for (i,j) in y_all], 'bo', markersize = 4)
            plt.plot(x, [j for (i,j) in y_all], 'bo', markersize = 4)
            #plt.plot((x,x),([i for (i,j) in y], [j for (i,j) in y]),c='black')

            plt.xlabel('Test Observation (X_{n+1})')
            plt.ylabel('Conformal Set Interval')
            plt.title('Variable Importance for feature \n ' + variable)
            plt.show()

        return conformal_sets_all, conformal_sets_j


    def difference_function(self, conformal_sets_all, conformal_sets_j, loss_type):

        if loss_type == 'IoU':   
            loss = []
            for i in range(len(conformal_sets_all)):
                tmp = (max(0, conformal_sets_all[i][0]-conformal_sets_j[i][0])+
                       max(0, conformal_sets_j[i][1]-conformal_sets_all[i][1]))/(max(conformal_sets_all[i][1], conformal_sets_j[i][1])-min(conformal_sets_all[i][0], conformal_sets_j[i][0]))
                loss.append(tmp)
        elif loss_type == 'basic':
            pass

        return loss

    def evaluate(self, conformal_set, true_y, method='acc'):
        
        true_y = true_y.tolist()
        if method == 'acc':
            eval = 0
            for i, interval in enumerate(conformal_set):
                eval += interval[0] <= true_y[i] <= interval[1]
            eval = eval/len(true_y)
        else:
            eval = 0

        return eval




def plot_ci(x, y, yerr, title):
    """
    Aux function used to plot results
    :param x: the variables investigated in the variable importance framework
    :param y: estimated feature importance value
    :param yerr: confidence interval (we assume in this )
    """
    tmp=[]
    for i, _ in enumerate(x):
        tmp_ci = [yerr[i][0]-y[i], y[i]-yerr[i][1]]
        tmp.append(tmp_ci)
    
    plt.figure(figsize=(12, 8))
    plt.errorbar(x, y, yerr=np.array(tmp).T, fmt='o')
    plt.xlabel('Variables')
    plt.ylabel('Variable Importance \n Quantification')
    plt.axline(xy1=(0, 0), slope=0, color='r', linestyle='--')
    plt.xticks(rotation=90)
    plt.title(title)
    plt.show()

