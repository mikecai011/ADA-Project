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
    scoring_function(model, x, y)
"""
def forwards_recursive_feature_selection(model_creation_method, x, y, scoring_function, stop_after=None, minimize=True):
    selected_features = []
    selected_feature_scores = []
    ban_list = []

    stop_after = stop_after if stop_after is not None else len(x.columns)
    stop_after = min(stop_after, len(x.columns))

    if minimize == True:
        r_scoring_function = scoring_function
    else:
        r_scoring_function = lambda model, x, y: -scoring_function(model, x, y)

    while len(selected_features) < stop_after:
        iteration_best = None
        iteration_min = float("inf")

        for feature in x.columns:
            if feature in selected_features:
                continue
            if feature in ban_list:
                continue

            current_features = selected_features + [feature]

            model = model_creation_method()
            try:
                model.fit(x[current_features], y)
            except:
                ban_list.append(feature)
                print("Ignoring", feature)
                continue
            score = r_scoring_function(model, x[current_features], y)

            if score < iteration_min:
                iteration_min = score
                iteration_best = feature

        selected_features = selected_features + [iteration_best]
        selected_feature_scores = selected_feature_scores + [iteration_min]
        print("Selected {0}".format(iteration_best))

    if minimize != True:
        selected_feature_scores = [-x for x in selected_feature_scores]

    print("WARNING: These features are ignored:", ban_list)
    return list(zip(selected_features, selected_feature_scores))
















