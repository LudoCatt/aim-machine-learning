from aim_machine_learning.model_evaluator import ModelEvaluator
import matplotlib.pyplot as plt

class ParametersTuner():
    def __init__(self, model_class, X, y, supported_eval_types, **output_path):
        self.model_class = model_class #istanzieremo un oggetto della classe del modello dato
        self.X = X
        self.y = y
        self.supported_eval_types = supported_eval_types
        self.output_path=output_path

    def tune_parameters(self, k_dict, eval_type, eval_obj, **params):
        self.eval_type = eval_type
        if eval_type not in self.supported_eval_types:
            raise NameError('Metodo di valutazione non supportato')

        minimo=float('inf')
        the_best_k=-1
        res=[]
        my_k=list(k_dict.keys())[0] 
        for k in k_dict[my_k]:
            mod_eval = ModelEvaluator(self.model_class, {my_k:k}, self.X, self.y)
            if eval_type == 'ttsplit':
                ub_mse = mod_eval.train_test_split_eval(eval_obj, params['test_proportion'])
            else:                   
                ub_mse = mod_eval.kfold_cv_eval(eval_obj, params['K'])
            res.append(ub_mse['mean']+ub_mse['std'])
            if res[-1]<minimo:
                minimo=res[-1]
                the_best_k=k

        prova=params.get('fig_name')
        if len(self.output_path)>0 and prova is not None:
            plt.figure()
            plt.plot(k_dict[my_k],res)
            plt.xlabel('k')
            plt.ylabel('Upper bound MSE')
            plt.savefig(self.output_path['output_path']+params['fig_name'])
# per risalire al path e dare il nome ho unito i due pezzi passati
# fig_name risulta uno degli elementi dei **params, infatti viene unito tutto
        return {my_k: the_best_k}

    
                
