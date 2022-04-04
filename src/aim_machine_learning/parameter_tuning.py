from aim_machine_learning.model_evaluator import ModelEvaluator
import matplotlib.pyplot as plt
import numpy as np

class ParametersTuner():
    def __init__(self, model_class, X, y, supported_eval_types, **output_path):
        self.model_class = model_class #istanzieremo oggetti della classe del modello dato
        self.X = X
        self.y = y
        self.supported_eval_types = supported_eval_types
        self.output_path = output_path

    def tune_parameters(self, param_dict, eval_type, eval_obj, **params):
        self.eval_type = eval_type
        if eval_type not in self.supported_eval_types:
            raise NameError('Metodo di valutazione non supportato')

        prova=params.get('fig_name') # servirà per fare i plot eventualmente
        if len(self.output_path)>0 and prova is not None:
            flag=True
        else: 
            flag=False

        minimo=float('inf') # se ci sono dati viene sovrascritto
        param_res={}
        res=[] # salvo i risultati per poter fare un plot alla fine 
        if len(list(param_dict.keys()))==1:
            my_k=list(param_dict.keys())[0] 
            for k in param_dict[my_k]:
                mod_eval = ModelEvaluator(self.model_class, {my_k:k}, self.X, self.y)
                if eval_type == 'ttsplit':
                    ub_mse = mod_eval.train_test_split_eval(eval_obj, params['test_proportion'])
                else:                   
                    ub_mse = mod_eval.kfold_cv_eval(eval_obj, params['K'])
                res.append(ub_mse['mean']+ub_mse['std'])
                if res[-1]<minimo:
                    minimo=res[-1]
                    param_res[my_k]=k

            if flag:
                plt.figure()
                plt.plot(param_dict[my_k],res)
                plt.xlabel(my_k)
                plt.ylabel('Upper bound MSE')
                plt.savefig(self.output_path['output_path']+params['fig_name'])

# vediamo la miglior combinazione dei due parametri
        elif len(param_dict.keys())==2:
            my_a=list(param_dict.keys())[0]
            my_b=list(param_dict.keys())[1]
            for b in param_dict[my_b]:
                for a in param_dict[my_a]:
                    mod_eval = ModelEvaluator(self.model_class, {my_a:a, my_b:b}, self.X, self.y)
                    if eval_type == 'ttsplit':
                        ub_mse = mod_eval.train_test_split_eval(eval_obj, params['test_proportion'])
                    else:                   
                        ub_mse = mod_eval.kfold_cv_eval(eval_obj, params['K'])
                    res.append(ub_mse['mean']+ub_mse['std'])
                    if res[-1]<minimo:
                        minimo=res[-1]
                        param_res[my_a]=a
                        param_res[my_b]=b
            res=np.array(res).reshape((len(param_dict[my_b]), len(param_dict[my_a])))
# considero che nel fare un disegno nel caso multivariato, b sia fissato e ci interessi la pendenza della retta
# disegno quindi a b fissato eventulmente sovrapponendo i grafici se si è interessati a più valori di b
            if flag:
                plt.figure()
                for i in range(len(param_dict[my_b])):
                    plt.plot(param_dict[my_a], res[i,:], label='b={}'.format(param_dict[my_b][i]))
                plt.legend(loc="upper left")
                plt.xlabel(my_a)
                plt.ylabel('Upper bound MSE')
                plt.savefig(self.output_path['output_path']+params['fig_name'])

        else:
            raise TypeError('Si è superato il numero massimo di parametri supportato (2)')
# per risalire al path e dare il nome ho unito i due pezzi passati
# fig_name risulta uno degli elementi dei **params, infatti viene unito tutto
        return param_res

    
                
