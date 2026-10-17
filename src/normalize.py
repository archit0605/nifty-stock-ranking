import numpy as np
import pandas as pd


def rank_to_unit_range(df, column):
    # rank within each date, smallest value gets rank 1, nan stays nan
    ranks = df.groupby("Date")[column].rank(method="average")

    # number of stocks with a real value on that date
    counts = df.groupby("Date")[column].transform("count")

    # map rank 1..n onto -1..1
    scaled = ((ranks - 1) / (counts - 1)) * 2 - 1

    # missing values go to the middle rank (0)
    scaled = scaled.fillna(0)
    return scaled


def rank_normalize(df, columns):
    result = df.copy()
    for column in columns:
        result[column + "_n"] = rank_to_unit_range(result, column)
    return result


def sector_relative(df, columns, min_stocks=3):
    result = df.copy()
    for column in columns:
        sector_group = result.groupby(["Date", "Sector"])[column]
        sector_median = sector_group.transform("median")
        sector_count = sector_group.transform("count")

        # value minus the median of its own sector on that date
        relative_value = result[column] - sector_median

        # sector too small for a trustworthy median, so set to nan
        relative_value = relative_value.where(sector_count >= min_stocks)

        # rank-normalize the relative values within each date
        temp = result[["Date"]].copy()
        temp["relative_value"] = relative_value
        result[column + "_secrel"] = rank_to_unit_range(temp, "relative_value")
    return result
