import pandas as pd


def get_decision_dates(panel):
    # use every date in the data (all tickers together) so everyone shares one grid
    all_dates = pd.Series(panel["Date"].unique())
    all_dates = all_dates.sort_values()

    # label each date with its year-month, e.g. 2019-01
    year_month = all_dates.dt.to_period("M")

    # the last trading date inside each month is that month's decision date
    decision_dates = all_dates.groupby(year_month).max()

    # check if the final month is complete
    # the month is complete only if the data reaches that month's last weekday
    last_date_in_data = all_dates.max()
    last_weekday_of_month = last_date_in_data + pd.offsets.BMonthEnd(0)
    if last_date_in_data < last_weekday_of_month:
        # data stops partway through the month, so drop that month
        decision_dates = decision_dates.iloc[:-1]

    # return a plain list-like of dates (DatetimeIndex)
    decision_dates = pd.DatetimeIndex(decision_dates.values, name="Date")
    return decision_dates


def make_month_end_prices(panel, decision_dates):
    # keep only rows that fall on a decision date
    is_decision_date = panel["Date"].isin(decision_dates)
    month_end_rows = panel[is_decision_date]

    # wide table: one row per decision date, one column per ticker
    month_end_prices = month_end_rows.pivot(index="Date", columns="Ticker", values="Adj_Close")
    month_end_prices = month_end_prices.sort_index()

    return month_end_prices


def make_forward_returns(month_end_prices, months_ahead):
    # each row is one month, so shifting by months_ahead rows moves months_ahead months
    # this is the only place a negative shift is allowed: labels look forward on purpose,
    # features must never do this
    future_prices = month_end_prices.shift(-months_ahead)

    # simple return from this decision date to the one months_ahead later
    forward_returns = future_prices / month_end_prices - 1

    return forward_returns
