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

def tune_params_with_count(fit_predict_function, train_df, feature_cols, param_grid, default_params, year):
    # returns the best params, how many candidates were tried and how many of them gave constant scores
    # no grid = nothing to tune
    if param_grid is None:
        return {}, 0, 0

    # inner validation year = the last year before `year`, inner train = everything before that
    inner_val_start = pd.Timestamp(str(year - 1) + "-01-01")
    inner_train_df = train_df[train_df["Date"] < inner_val_start]
    inner_val_df = train_df[train_df["Date"] >= inner_val_start]

    # the very first step has only one year of history, so there is nothing to tune on
    if len(inner_train_df) == 0:
        return default_params, 0, 0

    assert inner_train_df["Date"].max() < inner_val_df["Date"].min(), "inner train overlaps inner validation"

    best_ic = -np.inf
    best_params = None
    n_constant = 0
    for params in param_grid:
        scores = fit_predict_function(inner_train_df, inner_val_df, feature_cols, params)
        scored = inner_val_df[["Date", "Ticker", "fwd_1m"]].copy()
        scored["score"] = scores
        mean_ic = rank_ic_by_date(scored, "score").mean()

        # constant scores in every inner-validation month give an undefined ic, skip this candidate
        if np.isnan(mean_ic):
            n_constant = n_constant + 1
            continue

        if mean_ic > best_ic:
            best_ic = mean_ic
            best_params = params

    # if every setting gave a constant score (undefined ic), fall back to the default
    if best_params is None:
        return default_params, len(param_grid), n_constant

    return best_params, len(param_grid), n_constant


def tune_params(fit_predict_function, train_df, feature_cols, param_grid, default_params, year):
    # same as above, but only returns the best params
    best_params, n_tried, n_constant = tune_params_with_count(fit_predict_function, train_df, feature_cols,
                                                              param_grid, default_params, year)
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

        params, n_tried, n_constant = tune_params_with_count(fit_predict_function, train_df, feature_cols,
                                                             param_grid, default_params, year)
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
            "n_candidates_tried": n_tried,
            "n_constant_candidates": n_constant,
        })

    all_predictions = pd.concat(all_predictions).reset_index(drop=True)
    step_log = pd.DataFrame(step_rows)
    return all_predictions, step_log


# ---------- shared data setup (used by notebooks 06 and 07) ----------

# the momentum features are multiplied by these stock features and by these market-state variables
MOMENTUM_COLS = ["mom_1m_n", "mom_6_1_n", "mom_12_1_n"]
STOCK_PARTNER_COLS = ["retvol_63d_n", "amihud_21d_n", "abn_turnover_n"]
STATE_PARTNER_COLS = ["mkt_vol_63d_z", "mkt_ret_12m_z", "xs_dispersion_z"]


def prepare_model_data(model_table, keep_splits, labelled_only):
    # keep only the wanted splits (the warm-up rows are never kept unless asked for)
    keep_rows = model_table["split"].isin(keep_splits)
    if labelled_only:
        keep_rows = keep_rows & model_table["has_label_1m"]

    df = model_table[keep_rows].copy()
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # target = centered rank, the evaluation label stays fwd_1m
    df["y"] = df["fwd_1m_rank"] - 0.5

    # the 17 stock features and the state z-scores
    feature_cols = []
    state_z_cols = []
    for name in df.columns:
        if name.endswith("_n") or name.endswith("_secrel"):
            feature_cols.append(name)
        if name.endswith("_z"):
            state_z_cols.append(name)

    # state z-scores are nan early on (they need 12 months of history), 0 = neutral
    df[state_z_cols] = df[state_z_cols].fillna(0)

    return df, feature_cols, state_z_cols


def add_standard_interactions(df):
    # feature x feature and feature x state products used by M5, M6 and M7
    df, feature_x_feature_cols = add_interaction_columns(df, MOMENTUM_COLS, STOCK_PARTNER_COLS)
    df, feature_x_state_cols = add_interaction_columns(df, MOMENTUM_COLS, STATE_PARTNER_COLS)
    return df, feature_x_feature_cols, feature_x_state_cols


def build_model_registry(feature_cols, feature_x_feature_cols, feature_x_state_cols):
    # trees get the 17 features + 3 raw state columns and can find interactions themselves
    tree_cols = feature_cols + STATE_PARTNER_COLS

    # order here = simplicity order (M0 simplest, M9 most complex)
    model_registry = {
        "M0": {"label": "best single feature (mom_1m_n, sign from train ic)", "short": "single",
               "function": m0_fit_predict, "features": ["mom_1m_n"], "grid_options": None, "default": None},
        "M1": {"label": "equal-weight composite of 7 features with a prior", "short": "composite",
               "function": m1_fit_predict, "features": list(COMPOSITE_PRIORS.keys()), "grid_options": None, "default": None},
        "M2": {"label": "ols on 17 features", "short": "ols",
               "function": ols_fit_predict, "features": feature_cols, "grid_options": None, "default": None},
        "M3": {"label": "elastic net on 17 features", "short": "enet",
               "function": elastic_net_fit_predict, "features": feature_cols,
               "grid_options": ENET_GRID, "default": ENET_DEFAULT},
        "M4": {"label": "splines + ridge (nonlinear, no interactions)", "short": "splines",
               "function": spline_ridge_fit_predict, "features": feature_cols,
               "grid_options": RIDGE_GRID, "default": RIDGE_DEFAULT},
        "M5": {"label": "elastic net + feature x feature", "short": "enet+ff",
               "function": elastic_net_fit_predict, "features": feature_cols + feature_x_feature_cols,
               "grid_options": ENET_GRID, "default": ENET_DEFAULT},
        "M6": {"label": "elastic net + feature x state", "short": "enet+fs",
               "function": elastic_net_fit_predict, "features": feature_cols + feature_x_state_cols,
               "grid_options": ENET_GRID, "default": ENET_DEFAULT},
        "M7": {"label": "elastic net + both interaction sets", "short": "enet+both",
               "function": elastic_net_fit_predict,
               "features": feature_cols + feature_x_feature_cols + feature_x_state_cols,
               "grid_options": ENET_GRID, "default": ENET_DEFAULT},
        "M8": {"label": "random forest", "short": "rf",
               "function": random_forest_fit_predict, "features": tree_cols,
               "grid_options": RF_GRID, "default": RF_DEFAULT},
        "M9": {"label": "lightgbm", "short": "lgbm",
               "function": lightgbm_fit_predict, "features": tree_cols,
               "grid_options": LGBM_GRID, "default": LGBM_DEFAULT},
    }

    # turn each grid dictionary into the list of all combinations
    for name in model_registry:
        grid_options = model_registry[name]["grid_options"]
        if grid_options is None:
            model_registry[name]["grid"] = None
        else:
            model_registry[name]["grid"] = make_param_grid(grid_options)

    return model_registry
