from aim_machine_learning.base_regressor import Regressor

class NeighborRegressor(Regressor):
    def __init__(self, a, b):  # se non viene impostato diversamente, k=1 
        super().__init__()

    def fit(self, X_, y):
        self.X_train=X_
        self.y_train=y 

    def predict(self, X_): 
        pass

