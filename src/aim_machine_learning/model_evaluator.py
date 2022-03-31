from xml.dom import ValidationErr
import numpy as np

class ModelEvaluator():
    def __init__(self, model_class, params, X, y):
        self.model = model_class(**params)
        self.X = X
        self.y = y

    def train_test_split_eval(self, eval_obj, test_proportion):
        X_sample, X_train = np.split(self.X, [int(test_proportion*self.X.shape[0])])
        y_true, y_train = np.split(self.y, [int(test_proportion*self.y.shape[0])])
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_sample)
        return eval_obj(y_true=y_true, y_pred=y_pred)

    def kfold_cv_eval(self, eval_obj, K):

        if not isinstance(K,int):
            K=int

        A=self.X.shape[0]//K
        result={}

# purtroppo nonostante ininiti tentati non sono riuscita a gestire i casi estremi nel for
# perché non è possibile concatenare con un vettore vuoto
        self.model.fit(self.X[A:], self.y[A:])
        y_pred = self.model.predict(self.X[0:A])
        evaluations = eval_obj(y_true=self.y[0:A], y_pred=y_pred)
        result=self.sum_dict(result, evaluations)

        for k in range(1,K-1):
            X_train=np.concatenate((self.X[0:k*A],self.X[(k+1)*A:]))
            y_train=np.concatenate((self.y[0:k*A],self.y[(k+1)*A:]))
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(self.X[k*A:(k+1)*A])
            evaluations = eval_obj(y_true=self.y[k*A:(k+1)*A], y_pred=y_pred)
            result=self.sum_dict(result, evaluations)

        self.model.fit(self.X[:(K-1)*A], self.y[:(K-1)*A])
        y_pred = self.model.predict(self.X[(K-1)*A:])
        evaluations = eval_obj(y_true=self.y[(K-1)*A:], y_pred=y_pred)
        result=self.sum_dict(result, evaluations)

        for key, value in result.items():
            result[key]=round(value/K, 2)
        return result

    def sum_dict(self, somma, dic):
        for key, value in dic.items():
            somma[key] = somma.get(key, 0) + value
        return somma
        
        