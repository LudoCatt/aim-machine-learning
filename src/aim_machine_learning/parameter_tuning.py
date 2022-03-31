class ParametersTuner():
    def __init__(self, model_class, params, X, y):
        self.model = model_class(**params)
        self.X = X
        self.y = y