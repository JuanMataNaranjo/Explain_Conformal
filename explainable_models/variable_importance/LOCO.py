import random
import numpy as np

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

    def run_loco(self, variable, prop=0.7, loss_type='normal', bootstrap=1000):
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

        D1_X, D1_y, D2_X, D2_y = self.split(self.train_data, prop=prop)
        f1 = self.model.fit(D1_X, D1_y)
        f1_j = self.model.fit(D1_X.loc[:, D1_X.columns != variable], D1_y)
        pred_D2 = self.model.predict(f1, D2_X, softmax=False)
        pred_D2_j = self.model.predict(f1_j, D2_X.loc[:, D2_X.columns != variable], softmax=False)

        feature_importance = self.loss(true_y=D2_y, 
                                       pred_y=pred_D2, pred_y_j=pred_D2_j, type=loss_type)

        # Next thing is to estimate the confidence interval using bootstrap
        # TODO: Make sure this is correct (continue from here)
        median = []
        for i in range(bootstrap):

            idx = random.choices(range(len(D2_X)), len(D2_X))
            D2_X_boot = D2_X.loc[idx, D2_X.columns != variable]
            D2_y_boot = D2_y.loc[idx]

            pred_D2_j_boot = self.model.predict(f1_j, D2_X_boot, softmax=False)
            pred_D2_boot = pred_D2.loc[idx]

            median.append(np.median(self.loss(true_y=D2_y_boot, 
                                              pred_y=pred_D2_boot, pred_y_j=pred_D2_j_boot, 
                                              type=loss_type)))

        ci = self.confidence_interval()

        return feature_importance, ci


    def confidence_interval(self):

        pass
                

    def loss(self, true_y, pred_y, pred_y_j, type):
        """
        This method will compute different loss functions for the estimation of the feature importance variable. It will allow us 
        to implement multiple different loss functions depending on whether the variables are continuous or discrete
        """

        if type ==  'normal':
            loss_value = abs(true_y-pred_y_j)-abs(true_y-pred_y)

        return loss_value



    def split(self, data, prop=0.7):

        D1_index = random.sample(range(len(data)), int(len(data)*prop))

        indexes = range(len(data))
        D2_index = set(indexes).difference(set(D1_index))

        D1 = data.loc[D1_index]
        D2 = data.loc[D2_index]

        D1_X = D1.loc[:, D1.columns != self.y_feature]
        D1_y = D1.loc[:, self.y_feature]
        D2_X = D2.loc[:, D1.columns != self.y_feature]
        D2_y = D2.loc[:, self.y_feature]

        return D1_X, D1_y, D2_X, D2_y


class OldConformalLOCO():

    def __init__(self, model, train_data, y_feature):
        
        self.model = model
        self.train_data = train_data
        self.y_feature = y_feature


class NewConformalLOCO():

    def __init__(self, model, train_data, y_feature):
        
        self.model = model
        self.train_data = train_data
        self.y_feature = y_feature