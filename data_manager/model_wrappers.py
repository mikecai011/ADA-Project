from sklearn.preprocessing import PowerTransformer, MinMaxScaler
import pandas as pd

class NormalizingRegressionModel(object):
    def __init__(self, base_model, normalization_method, y_normalization_method = None):
        self.base_model = base_model
        self.normalization_method = normalization_method
        self.y_normalization_method = y_normalization_method if y_normalization_method else normalization_method
        self.x_normalizer = None
        self.y_normalizer = None

        self.columns = None

    def __normalize_x(self, x):
        normalized = self.x_normalizer.transform(x)
        return pd.DataFrame(normalized, index=x.index, columns=self.columns)

    def __normalize_y(self, y):
        normalized = self.y_normalizer.transform(y.to_frame()).reshape((-1,))
        return pd.Series(normalized, index=y.index)

    def __inverse_normalize_x(self, x, index):
        unnormalized = self.x_normalizer.inverse_transform(x)
        return pd.DataFrame(unnormalized, index=index, columns=self.columns)

    def __inverse_normalize_y(self, y, index):
        unnormalized = self.y_normalizer.inverse_transform(y.reshape((-1,1))).reshape((-1,))
        return pd.Series(unnormalized, index=index)

    def fit(self, x, y):
        if self.x_normalizer is None:
            if self.normalization_method == "scale":
                self.x_normalizer = MinMaxScaler()
                self.y_normalizer = MinMaxScaler()

            else:
                self.x_normalizer = PowerTransformer(self.normalization_method)
                self.y_normalizer = PowerTransformer(self.normalization_method)

            self.x_normalizer.fit(x)
            self.y_normalizer.fit(y.to_frame())

            self.columns = x.columns

        nx = self.__normalize_x(x)
        ny = self.__normalize_y(y)
        return self.base_model.fit(nx, ny)

    def predict(self, x):
        nx = self.__normalize_x(x)
        res = self.base_model.predict(nx)
        return self.__inverse_normalize_y(res, x.index)

    def score(self, x, y, sample_weight = None):
        nx = self.__normalize_x(x)
        ny = self.__normalize_y(y)
        return self.base_model.score(nx, ny, sample_weight)

class SampleFilteringRegressionModel(object):
    def __init__(self, base_model):
        self.base_model = base_model

    def __get_intersection(self, x, y):
        intersection = x.index.intersection(y.index)
        return x.loc[intersection, :], y.loc[intersection]

    def fit(self, x, y):
        x, y, = self.__get_intersection(x, y)
        return self.base_model.fit(x, y)

    def predict(self, x):
        return self.base_model.predict(x)

    def score(self, x, y, sample_weight = None):
        x, y, = self.__get_intersection(x, y)
        return self.base_model.score(x, y, sample_weight)

class ColumnFilteringRegressionModel(object):
    def __init__(self, base_model):
        self.base_model = base_model
        self.columns = None

    def fit(self, x, y):
        if self.columns is None:
            temp_x = x.copy()
            temp_x = temp_x[temp_x.columns[temp_x.var() != 0]]
            self.columns = temp_x.columns

        x = x[self.columns]
        return self.base_model.fit(x, y)

    def predict(self, x):
        x = x[self.columns]
        return self.base_model.predict(x)

    def score(self, x, y, sample_weight = None):
        x = x[self.columns]
        return self.base_model.score(x, y, sample_weight)


def test():
    from sklearn.neural_network import MLPRegressor
    import numpy as np
    # x = pd.DataFrame([
    #     [1,2,1],
    #     [4,5,0],
    #     [7,0,9],
    #     [0,0,0],
    #     [2,-4,9],
    #     [3,1,1]
    # ], index=['a', 'b', 'c', 'd', 'e', 'f'])
    x = pd.DataFrame(np.random.random_sample((10000, 5))*30-15)
    y = x.iloc[1:, 0]/3 + x.iloc[1:, 1]/2 + x.iloc[1:, 2]/4
    y = y.sample(frac=1)
    test_x = pd.DataFrame([
        [1, 1, 0, 1, 2],
        [2, 0, 9, 3, 1],
        [3, 3, 0, 3, 3]
    ], index=['1', '2', '3'])

    def create_model():
        regressor = MLPRegressor(hidden_layer_sizes=(5, 4), activation='relu', solver='adam')
        regressor = ColumnFilteringRegressionModel(regressor)
        regressor = NormalizingRegressionModel(
            regressor,
            "scale"
        )
        regressor = SampleFilteringRegressionModel(regressor)
        return regressor

    regressor = create_model()
    regressor.fit(x, y)
    print(regressor.predict(test_x))

    from feature_selection import get_features_by_score, forwards_recursive_feature_selection
    from feature_selection import FeatureScoringFunctions, ModelScoringFunctions
    print("Via permuatation:")
    fbs = get_features_by_score(
        regressor, x, y,
        FeatureScoringFunctions.get_permutation_score(ModelScoringFunctions.internal_score)
    )
    for fs in fbs:
        print(fs)

    print("Via recursion:")
    fbs = forwards_recursive_feature_selection(
        create_model,
        x,
        y,
        ModelScoringFunctions.internal_score,
        stop_after=10,
        minimize=False
    )
    for fs in fbs:
        print(fs)


if __name__ == "__main__":
    test()