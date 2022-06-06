import copy
import numpy as np

def IOU(conformal_sets_all, conformal_sets_j): 
    loss = []
    for i in range(len(conformal_sets_all)):
        tmp = (max(0, conformal_sets_all[i][0]-conformal_sets_j[i][0])+
                max(0, conformal_sets_j[i][1]-conformal_sets_all[i][1]))/(max(conformal_sets_all[i][1], conformal_sets_j[i][1])-min(conformal_sets_all[i][0], conformal_sets_j[i][0]))
        loss.append(tmp)
    return sum(loss)/len(conformal_sets_all)


def full_conformal(model, data_model, conformal_class, alpha=0.05, evaluate=True):

    model_full = model
    model_full.fit(data_model.train_data_X, data_model.train_data_y)

    conformal_predictor = conformal_class(alpha=alpha)
    lambda_all = conformal_predictor.calibrate(data_X=data_model.calib_data_X, data_y=data_model.calib_data_y, 
                                               model=model_full, rand_state=None)
    pred_all = conformal_predictor.predict(data_model.test_data_X, model=model_full, lambda_conformal=lambda_all)

    if evaluate:
        cov, size = conformal_predictor.evaluate(pred_all, data_model.test_data_y)

        return pred_all, cov, size, copy.copy(model_full)
    else:
        return pred_all, lambda_all, copy.copy(model_full)


def train_twice_conformal(model, modified_data, data_model, conformal_class, alpha=0.05, evaluate=True):

    model_j_2 = model
    model_j_2.fit(modified_data['train'], data_model.train_data_y)

    conformal_predictor_j = conformal_class(alpha=alpha)

    lambda_j_2 = conformal_predictor_j.calibrate(data_X=modified_data['calib'], data_y=data_model.calib_data_y, 
                                                            model=model_j_2, rand_state=None)
    pred_j_2 = conformal_predictor_j.predict(modified_data['test'], model=model_j_2, lambda_conformal=lambda_j_2)

    if evaluate:
        cov_j_2, size_j_2 = conformal_predictor_j.evaluate(pred_j_2, data_model.test_data_y)
        return pred_j_2, cov_j_2, size_j_2
    else:
        return pred_j_2, lambda_j_2

def train_once_conformal(model, modified_data, data_model, conformal_class, alpha=0.05, evaluate=True):

    conformal_predictor_j = conformal_class(alpha=alpha)

    lambda_j_1 = conformal_predictor_j.calibrate(data_X=modified_data['calib'], data_y=data_model.calib_data_y, 
                                                        model=model, rand_state=None)
    pred_j_1 = conformal_predictor_j.predict(modified_data['test'], model=model, lambda_conformal=lambda_j_1)

    if evaluate:
        cov_j_1, size_j_1 = conformal_predictor_j.evaluate(pred_j_1, data_model.test_data_y)
        return pred_j_1, cov_j_1, size_j_1
    else:
        return pred_j_1, lambda_j_1