import numpy as np


class FeatureScoringFunctions(object):
    @staticmethod
    def get_permutation_score(base_scorer):
        def internal(model, x, y, feature):
            min_score = 1
            for i in range(3):
                tempx = x.copy()
                tempx.loc[:, feature] = tempx.loc[:, feature].sample(frac=1).to_list()
                min_score = min(min_score,
                                base_scorer(model, tempx, y))
            return min_score
        return internal


class ModelScoringFunctions(object):
    @staticmethod
    def internal_score(model, x, y):
        return model.score(x, y)


def get_features_by_score(model, x, y, feature_scoring_function):
    features_to_scores = []
    for feature in x.columns:
        features_to_scores.append((
            feature,
            feature_scoring_function(model, x, y, feature)
        ))

    features_to_scores = sorted(features_to_scores, key=lambda x: x[1])
    return features_to_scores

"""
    Function is supposed to MINIMIZE
    THIS FUNCTION IS NOT IN USE
    scoring_function(model, x, y)
"""
# def forwards_recursive_feature_selection(model_creation_method, x, y, scoring_function, stop_after=None):
#     selected_features = []
#     selected_feature_scores = []
#
#     stop_after = stop_after if stop_after is not None else len(x.columns)
#
#     while len(selected_features) < stop_after:
#
#         for feature in x.columns:
#             if feature in selected_features:
#                 continue
