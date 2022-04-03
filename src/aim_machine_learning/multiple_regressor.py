from aim_machine_learning.base_regressor import Regressor
import numpy as np

class MultipleRegressor(Regressor):
    def __init__(self, a, b): 
        super().__init__()
        self.a = a
        self.b = b

    def fit(self, X, y): # il modello non si addestra ma sfrutta ParametersTuner
        self.X_train=X
        self.y_train=y 

    def predict(self, X): 
        a=np.array(self.a).reshape(-1) # lo rendo un vettore monodimensionale
        return (np.dot(X,a)+self.b).reshape(-1) # sfrutto il prodotto scalare

    def __add__(self, Mul_Reg2):
        return MultipleRegressor([[self.a], [Mul_Reg2.a]], self.b+Mul_Reg2.b)
# forzato vettore di vettori per far combaciare gli otputs


