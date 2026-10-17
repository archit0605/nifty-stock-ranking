import numpy as np
import pandas as pd

from src.labels import make_month_end_prices


def make_wide(panel, column):
    # one row per daily date, one column per ticker
    wide = panel.pivot(index="Date", columns="Ticker", values=column)
    wide = wide.sort_index()
    return wide


def make_market_return(daily_returns_wide):
    # equal-weight market: average return across all tickers on each day
    market_return = daily_returns_wide.mean(axis=1)
    market_return.name = "market_return"
    return market_return


def make_momentum_features(month_end_prices):
    # month_end_prices has one row per decision date, so shift(k) means k months back
    # shift(1) looks backward: price at the previous decision date (t-1)
    price_1m_ago = month_end_prices.shift(1)
    # shift(6) looks backward: price 6 decision dates ago (t-6)
    price_6m_ago = month_end_prices.shift(6)
    # shift(7) looks backward: price 7 decision dates ago (t-7)
    price_7m_ago = month_end_prices.shift(7)
    # shift(12) looks backward: price 12 decision dates ago (t-12)
    price_12m_ago = month_end_prices.shift(12)

    # last month's return, t-1 to t
    mom_1m = month_end_prices / price_1m_ago - 1

    # t-6 to t-1, skips the most recent month
    mom_6_1 = price_1m_ago / price_6m_ago - 1

    # t-12 to t-1, skips the most recent month
    mom_12_1 = price_1m_ago / price_12m_ago - 1

    # change in momentum: recent half (t-6 to t-1) minus older half (t-12 to t-7)
    older_half_return = price_7m_ago / price_12m_ago - 1
    chmom = mom_6_1 - older_half_return

    momentum_features = {
        "mom_1m": mom_1m,
        "mom_6_1": mom_6_1,
        "mom_12_1": mom_12_1,
        "chmom": chmom,
    }
    return momentum_features


def make_beta_and_idiovol(daily_returns_wide, market_return, window):
    beta_table = pd.DataFrame(index=daily_returns_wide.index)
    idiovol_table = pd.DataFrame(index=daily_returns_wide.index)

    # rolling variance of the market, only uses the last `window` days up to today
    market_var = market_return.rolling(window, min_periods=window).var()

    for ticker in daily_returns_wide.columns:
        stock_return = daily_returns_wide[ticker]

        # beta = cov(stock, market) / var(market) inside the same window
        stock_market_cov = stock_return.rolling(window, min_periods=window).cov(market_return)
        beta = stock_market_cov / market_var

        # idiovol = std of (stock - beta * market) inside the same window, with that window's beta
        # math shortcut: that variance equals var(stock) - beta^2 * var(market)
        stock_var = stock_return.rolling(window, min_periods=window).var()
        idio_var = stock_var - beta * beta * market_var
        idiovol = np.sqrt(idio_var)

        beta_table[ticker] = beta
        idiovol_table[ticker] = idiovol

    return beta_table, idiovol_table


def make_daily_window_features(daily_returns_wide, market_return, decision_dates):
    # every rolling window ends on (and includes) the current day, never later days
    retvol_21d = daily_returns_wide.rolling(21, min_periods=21).std()
    retvol_63d = daily_returns_wide.rolling(63, min_periods=63).std()
    maxret_21d = daily_returns_wide.rolling(21, min_periods=21).max()
    beta_252d, idiovol_252d = make_beta_and_idiovol(daily_returns_wide, market_return, 252)

    daily_features = {
        "retvol_21d": retvol_21d,
        "retvol_63d": retvol_63d,
        "maxret_21d": maxret_21d,
        "beta_252d": beta_252d,
        "idiovol_252d": idiovol_252d,
    }

    # keep only the decision-date rows, so each value uses data up to that date
    sampled_features = {}
    for name in daily_features:
        sampled_features[name] = daily_features[name].loc[decision_dates]

    return sampled_features


def make_liquidity_features(turnover_wide, daily_returns_wide, decision_dates):
    # rolling windows below all end today and only look backward
    mean_turnover_21d = turnover_wide.rolling(21, min_periods=21).mean()
    std_turnover_21d = turnover_wide.rolling(21, min_periods=21).std()
    mean_turnover_252d = turnover_wide.rolling(252, min_periods=252).mean()

    # log of the 21 day average rupees traded, puts big and small stocks on a similar scale
    log_turnover_21d = np.log(mean_turnover_21d)

    # how much turnover jumps around compared with its own average
    turnover_cv_21d = std_turnover_21d / mean_turnover_21d

    # last month's turnover compared with the last year, above 1 means unusually busy
    abn_turnover = mean_turnover_21d / mean_turnover_252d

    # amihud = price move per rupee traded, turnover of 0 would divide by zero so make it nan
    turnover_no_zero = turnover_wide.replace(0, np.nan)
    daily_amihud = daily_returns_wide.abs() / turnover_no_zero
    amihud_21d = daily_amihud.rolling(21, min_periods=21).mean() * 1e9

    daily_features = {
        "log_turnover_21d": log_turnover_21d,
        "turnover_cv_21d": turnover_cv_21d,
        "abn_turnover": abn_turnover,
        "amihud_21d": amihud_21d,
    }

    # keep only the decision-date rows
    sampled_features = {}
    for name in daily_features:
        sampled_features[name] = daily_features[name].loc[decision_dates]

    return sampled_features


def make_expanding_zscore(series, min_months=12):
    # mean and std use only the current and past values, never future ones
    expanding_mean = series.expanding(min_periods=min_months).mean()
    expanding_std = series.expanding(min_periods=min_months).std()
    zscore = (series - expanding_mean) / expanding_std
    return zscore


def make_state_variables(market_return, mom_1m_wide, decision_dates):
    # market level starts at 1 and grows by each day's equal-weight market return
    market_level = (1 + market_return.fillna(0)).cumprod()
    level_at_decisions = market_level.loc[decision_dates]

    # shift(1) looks backward: market level at the previous decision date
    level_1m_ago = level_at_decisions.shift(1)
    # shift(12) looks backward: market level 12 decision dates ago
    level_12m_ago = level_at_decisions.shift(12)

    # daily market volatility over the last 63 days, window ends today
    market_vol_63d = market_return.rolling(63, min_periods=63).std()

    state = pd.DataFrame(index=decision_dates)
    state["mkt_ret_1m"] = level_at_decisions / level_1m_ago - 1
    state["mkt_ret_12m"] = level_at_decisions / level_12m_ago - 1
    state["mkt_vol_63d"] = market_vol_63d.loc[decision_dates]

    # how spread out the 30 stocks' last-month returns are on that date
    state["xs_dispersion"] = mom_1m_wide.std(axis=1)

    # add an expanding z-score version of each one
    raw_names = ["mkt_ret_1m", "mkt_ret_12m", "mkt_vol_63d", "xs_dispersion"]
    for name in raw_names:
        state[name + "_z"] = make_expanding_zscore(state[name])

    return state


def wide_dict_to_long(feature_dict):
    # turns {name: wide table} into one long table with a column per feature
    long_table = None
    for name in feature_dict:
        one_feature_long = feature_dict[name].reset_index().melt(id_vars="Date", var_name="Ticker", value_name=name)
        if long_table is None:
            long_table = one_feature_long
        else:
            long_table = pd.merge(long_table, one_feature_long, on=["Date", "Ticker"])

    long_table = long_table.sort_values(["Ticker", "Date"])
    long_table = long_table.reset_index(drop=True)
    return long_table


def build_all_features(panel, decision_dates):
    # runs every feature function in one go, used by the leakage test
    month_end_prices = make_month_end_prices(panel, decision_dates)
    daily_returns_wide = make_wide(panel, "Daily_Return")
    turnover_wide = make_wide(panel, "Turnover")
    market_return = make_market_return(daily_returns_wide)

    momentum_features = make_momentum_features(month_end_prices)
    daily_window_features = make_daily_window_features(daily_returns_wide, market_return, decision_dates)
    liquidity_features = make_liquidity_features(turnover_wide, daily_returns_wide, decision_dates)

    all_features = {}
    all_features.update(momentum_features)
    all_features.update(daily_window_features)
    all_features.update(liquidity_features)

    stock_features = wide_dict_to_long(all_features)
    state_variables = make_state_variables(market_return, momentum_features["mom_1m"], decision_dates)

    return stock_features, state_variables
