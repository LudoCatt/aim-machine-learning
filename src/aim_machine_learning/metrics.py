import numpy as np

class Evaluator():

    def __init__(self, supported_metrics, **params):
        self.metriche = supported_metrics
    
    def set_metric(self, new_metric):
        self.metrica = new_metric
        if self.metrica not in self.metriche:
            raise NameError('La metrica non è supportata')
        return self
    
    def __repr__(self): 
        return 'Current metric is {}'.format(self.metrica)

    def __call__(self, y_true, y_pred):
        res={}
        if self.metrica=='mse':
            res['mean']=round(np.mean((y_true - y_pred)**2),2)
            res['std']=round(np.std((y_true - y_pred)**2),2)
        elif self.metrica=='mae':
            res['mean']=round(np.mean(abs(y_true - y_pred)),2)
            res['std']=round(np.std(abs(y_true - y_pred)),2)
        elif self.metrica=='corr':
            res['corr']=round(np.corrcoef(y_true, y_pred)[0,1],2)
        return res
