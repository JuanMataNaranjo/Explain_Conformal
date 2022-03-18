from abc import ABCMeta, abstractmethod
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


class BaseModel(metaclass=ABCMeta):


    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError()



class SVM(BaseModel):
    """ SVM Model """

    def __init__(self, kernel='linear', C=1, probability=True, verbose=False):

        self.model = SVC(kernel=kernel, C=C, probability=probability, verbose=verbose)

    def fit(self, train_X, train_y):
        """
        Fit the model on the data
        """
        return self.model.fit(train_X, train_y)

    def evaluate(self, model, test_X, test_y):
        
        predictions = model.predict(test_X)
        accuracy = model.score(test_X, test_y)
        cm = confusion_matrix(test_y, predictions)

        return accuracy, cm

    def predict(self, model, data, softmax=True):

        if softmax:
            pred = model.predict_proba(data)
        else:
            pred = model.predict(data)

        return pred


class LogisticReg(BaseModel):
    """ Logistic Regression """

    def __init__(self, C=1, verbose=0, max_iter=100):

        self.model = LogisticRegression(C=C, verbose=verbose, max_iter=max_iter)

    def fit(self, train_X, train_y):
        """
        Fit the model on the data
        """
        return self.model.fit(train_X, train_y)

    def evaluate(self, model, test_X, test_y):
        
        predictions = model.predict(test_X)
        accuracy = model.score(test_X, test_y)
        cm = confusion_matrix(test_y, predictions)

        return accuracy, cm

    def predict(self, model, data, softmax=True):

        if softmax:
            pred = model.predict_proba(data)
        else:
            pred = model.predict(data)

        return pred
