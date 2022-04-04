from aim_machine_learning.base_regressor import Regressor
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

class NeighborRegressor(Regressor):

    def __init__(self, k=1, **params):  # se non viene impostato diversamente, k=1 
        super().__init__(**params) 
        self.k=k 
        self.X_train = None # se prima non  fa fit non può fare predict
        self.y_train = None
        
    def fit(self, X, y):
        self.X_train=X
        self.y_train=y
        
    def predict(self, X_test): 
        
        if self.X_train is None or self.y_train is None: 
            raise NameError('Non è stato allenato, non è possibile fare previsioni')
        
        n_test=X_test.shape[0]
        n_train=self.X_train.shape[0] 
        predictions=np.zeros(n_test)

# si osservi che senza try il codice non funzionerebbe nel caso in cui X_test sia monodimensionale  
        try:      
            for i in range(n_test):
                distances=np.zeros(n_train)
                for j in range(n_train):
                    distances[j]=self.__distance(self.X_train[j, :], X_test[i, :])    
                indices=np.argpartition(distances, self.k)[0:self.k]   
                predictions[i]=np.mean(self.y_train[indices])
        except IndexError:
            for i in range(n_test):
                distances=np.zeros(n_train)
                for j in range(n_train):
                    distances[j]=(self.X_train[j]-X_test[i])**2   
                indices=np.argpartition(distances, self.k)[0:self.k]   
                predictions[i]=np.mean(self.y_train[indices])
        
        return predictions
            
    def __distance(self, x1, x2): # il metodo può essere chiamato solo dall'interno
        return np.sum((x1-x2)**2)


class MySklearnNeighborRegressor(KNeighborsRegressor, Regressor):
    pass
# prende tutto ciò di cui ha bisogno dai suoi genitori
