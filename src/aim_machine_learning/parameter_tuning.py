class ParametersTuner():
    def __init__(self, model_class, X, y, supported_eval_types):
        self.model = model_class()
        self.X = X
        self.y = y
        self.supported_eval_types=supported_eval_types

    def tune_parameters(self, k_dict, eval_type, eval_obj, **params):
        self.eval_type = eval_type
        
