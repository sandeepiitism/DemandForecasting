
class Prediction_model:

    def __init__(self, series_df, detail_df, fh, model_gov,tune_gov,params):
    # def __init__(self, series_df, fh, model_gov,tune_gov):
        self.series_df = series_df
        self.scaled_series_df = series_df
        self.detail_df = detail_df
        self.fh = fh
        self.model_gov = model_gov
        self.tune_gov = tune_gov
        self.params = params

    def predict(self):
        pass

    def to_long(self, combined_df):
        pass