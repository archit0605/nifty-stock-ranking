import pandas as pd


def load_data(path):
    # read the csv, parse_dates makes Date actual dates instead of text
    df = pd.read_csv(path, parse_dates=["Date"])

    # put each stock's rows together, oldest date first
    df = df.sort_values(["Ticker", "Date"])

    # after sorting the index is jumbled, so number the rows 0, 1, 2... again
    df = df.reset_index(drop=True)

    return df
