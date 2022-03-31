class ParametersTuner():
    def __init__(self, model_class, X, y, supported_eval_types):
        self.model = model_class(**params)
        self.X = X
        self.y = y