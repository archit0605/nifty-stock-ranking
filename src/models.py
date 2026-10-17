import itertools

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from src.evaluate import rank_ic_by_date

# literature direction for the equal-weight composite (M1)
COMPOSITE_PRIORS = {
    "mom_1m_n": -1,
    "mom_6_1_n": 1,
    "mom_12_1_n": 1,
    "retvol_21d_n": -1,
    "retvol_63d_n": -1,
    "maxret_21d_n": -1,
    "idiovol_252d_n": -1,
}

# small hyperparameter grids (a list of values for each setting)
ENET_GRID = {"alpha": [0.001, 0.01, 0.1], "l1_ratio": [0.2, 0.5, 0.8]}
RIDGE_GRID = {"alpha": [1, 10, 100]}
RF_GRID = {"max_depth": [3, 5], "min_samples_leaf": [20, 50], "n_estimators": [300], "random_state": [42]}
LGBM_GRID = {"num_leaves": [7, 15], "learning_rate": [0.03], "n_estimators": [200],
             "min_child_samples": [30, 60], "subsample": [0.8], "colsample_bytree": [0.8],
             "random_state": [42]}

# used for the first walk-forward year (2021) where there is no earlier year to tune on
# each one is a middle-of-the-grid choice (elastic net alpha 0.1 would shrink every coefficient to 0)
ENET_DEFAULT = {"alpha": 0.01, "l1_ratio": 0.5}
RIDGE_DEFAULT = {"alpha": 100}
RF_DEFAULT = {"max_depth": 3, "min_samples_leaf": 50, "n_estimators": 300, "random_state": 42}
LGBM_DEFAULT = {"num_leaves": 7, "learning_rate": 0.03, "n_estimators": 200, "min_child_samples": 60,
                "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42}


def make_param_grid(grid_options):
    # turns {"alpha": [1, 2], "l1": [0.1, 0.5]} into a list of all 4 combinations
    names = list(grid_options.keys())
    value_lists = []
    for name in names:
        value_lists.append(grid_options[name])

    param_grid = []
    for combination in itertools.product(*value_lists):
        param_grid.append(dict(zip(names, combination)))
    return param_grid


def add_interaction_columns(df, left_cols, right_cols):
    # multiply every left column by every right column, row by row
    result = df.copy()
    new_names = []
    for left in left_cols:
        for right in right_cols:
            new_name = left + "_x_" + right
            result[new_name] = result[left] * result[right]
            new_names.append(new_name)
    return result, new_names


# ---------- estimators (unfitted) ----------

def make_elastic_net(params):
    # scale first so the penalty treats every column fairly (the scaler only sees the training rows)
    model = Pipeline([
        ("scale", StandardScaler()),
        ("enet", ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"], max_iter=10000)),
    ])
    return model


def make_spline_ridge(params):
    # SplineTransformer bends each feature on its own, ridge then fits a straight line on those curves
    model = Pipeline([
        ("spline", SplineTransformer(n_knots=4, degree=3)),
        ("ridge", Ridge(alpha=params["alpha"])),
    ])
    return model


def make_random_forest(params):
    model = RandomForestRegressor(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                  min_samples_leaf=params["min_samples_leaf"],
                                  random_state=params["random_state"], n_jobs=-1)
    return model


def make_lightgbm(params):
    # subsample_freq=1 is needed, otherwise lightgbm ignores the subsample setting
    model = lgb.LGBMRegressor(num_leaves=params["num_leaves"], learning_rate=params["learning_rate"],
                              n_estimators=params["n_estimators"], min_child_samples=params["min_child_samples"],
                              subsample=params["subsample"], subsample_freq=1,
                              colsample_bytree=params["colsample_bytree"], random_state=params["random_state"],
                              importance_type="gain", verbose=-1)
    return model


# ---------- fit + predict functions, one per model type ----------
# every one takes (train_df, predict_df, feature_cols, params) and returns one score per row of predict_df

def sklearn_fit_predict(model, train_df, predict_df, feature_cols):
    model.fit(train_df[feature_cols], train_df["y"])
    return model.predict(predict_df[feature_cols])


def m0_fit_predict(train_df, predict_df, feature_cols, params):
    # one feature, its sign comes from the train-only rank ic
    feature = feature_cols[0]
    train_ic = rank_ic_by_date(train_df, feature, "fwd_1m").mean()
    sign = np.sign(train_ic)
    return sign * predict_df[feature].to_numpy()


def m1_fit_predict(train_df, predict_df, feature_cols, params):
    # nothing to fit: average of the features, each one multiplied by its literature direction
    total = np.zeros(len(predict_df))
    for feature in feature_cols:
        total = total + COMPOSITE_PRIORS[feature] * predict_df[feature].to_numpy()
    return total / len(feature_cols)


def ols_fit_predict(train_df, predict_df, feature_cols, params):
    return sklearn_fit_predict(LinearRegression(), train_df, predict_df, feature_cols)


def elastic_net_fit_predict(train_df, predict_df, feature_cols, params):
    return sklearn_fit_predict(make_elastic_net(params), train_df, predict_df, feature_cols)


def spline_ridge_fit_predict(train_df, predict_df, feature_cols, params):
    return sklearn_fit_predict(make_spline_ridge(params), train_df, predict_df, feature_cols)


def random_forest_fit_predict(train_df, predict_df, feature_cols, params):
    return sklearn_fit_predict(make_random_forest(params), train_df, predict_df, feature_cols)


def lightgbm_fit_predict(train_df, predict_df, feature_cols, params):
    return sklearn_fit_predict(make_lightgbm(params), train_df, predict_df, feature_cols)


# ---------- walk-forward ----------

def tune_params(fit_predict_function, train_df, feature_cols, param_grid, default_params, year):
    # no grid = nothing to tune
    if param_grid is None:
        return {}

    # inner validation year = the last year before `year`, inner train = everything before that
    inner_val_start = pd.Timestamp(str(year - 1) + "-01-01")
    inner_train_df = train_df[train_df["Date"] < inner_val_start]
    inner_val_df = train_df[train_df["Date"] >= inner_val_start]

    # the very first step has only one year of history, so there is nothing to tune on
    if len(inner_train_df) == 0:
        return default_params

    assert inner_train_df["Date"].max() < inner_val_df["Date"].min(), "inner train overlaps inner validation"

    best_ic = -np.inf
    best_params = None
    for params in param_grid:
        scores = fit_predict_function(inner_train_df, inner_val_df, feature_cols, params)
        scored = inner_val_df[["Date", "Ticker", "fwd_1m"]].copy()
        scored["score"] = scores
        mean_ic = rank_ic_by_date(scored, "score").mean()
        if mean_ic > best_ic:
            best_ic = mean_ic
            best_params = params

    # if every setting gave a constant score (undefined ic), fall back to the default
    if best_params is None:
        return default_params

    return best_params


def walk_forward_predict(df, fit_predict_function, feature_cols, param_grid=None, default_params=None,
                         start_year=2021, end_year=2024):
    all_predictions = []
    step_rows = []

    for year in range(start_year, end_year + 1):
        year_start = pd.Timestamp(str(year) + "-01-01")
        next_year_start = pd.Timestamp(str(year + 1) + "-01-01")

        # train on everything before this year, predict every row of this year
        train_df = df[df["Date"] < year_start]
        predict_df = df[(df["Date"] >= year_start) & (df["Date"] < next_year_start)]

        # no lookahead: training data must end before the first prediction date
        assert train_df["Date"].max() < predict_df["Date"].min(), "train dates reach into the prediction year"

        params = tune_params(fit_predict_function, train_df, feature_cols, param_grid, default_params, year)
        scores = fit_predict_function(train_df, predict_df, feature_cols, params)

        predictions = predict_df[["Date", "Ticker"]].copy()
        predictions["score"] = scores
        all_predictions.append(predictions)

        step_rows.append({
            "year": year,
            "n_train": len(train_df),
            "n_predict": len(predict_df),
            "max_train_date": train_df["Date"].max(),
            "min_predict_date": predict_df["Date"].min(),
            "params": params,
        })

    all_predictions = pd.concat(all_predictions).reset_index(drop=True)
    step_log = pd.DataFrame(step_rows)
    return all_predictions, step_log
