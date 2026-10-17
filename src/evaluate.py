import numpy as np
import pandas as pd


def spearman_corr(first_values, second_values):
    # spearman correlation = normal correlation of the ranks
    first_ranks = first_values.rank()
    second_ranks = second_values.rank()
    return first_ranks.corr(second_ranks)


def rank_ic_by_date(df, score_col, label_col="fwd_1m"):
    # one spearman correlation per date, between the score and the label of the 30 stocks
    ic_values = {}
    for date, one_month in df.groupby("Date"):
        ic_values[date] = spearman_corr(one_month[score_col], one_month[label_col])

    ic_series = pd.Series(ic_values)
    ic_series.index.name = "Date"
    return ic_series


def ic_summary(ic_series):
    # months where the ic could not be computed are skipped
    ic_values = ic_series.dropna()
    n_months = len(ic_values)

    mean_ic = ic_values.mean()
    std_ic = ic_values.std()

    # a std of 0 means every month had the same ic (e.g. the perfect score), icir is undefined
    if std_ic == 0:
        icir = np.nan
        t_stat = np.nan
    else:
        icir = mean_ic / std_ic
        t_stat = mean_ic / (std_ic / np.sqrt(n_months))

    pct_positive = (ic_values > 0).mean() * 100

    summary = {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "icir": icir,
        "t_stat": t_stat,
        "pct_positive": pct_positive,
        "n_months": n_months,
    }
    return summary


def assign_buckets(df, score_col, n_buckets=5):
    # rank inside each date, ties are broken by row order so every bucket has the same size
    ranks = df.groupby("Date")[score_col].rank(method="first")
    stocks_per_date = df.groupby("Date")[score_col].transform("count")

    # ranks 1-6 -> bucket 1, 7-12 -> bucket 2, ... (for 30 stocks and 5 buckets)
    buckets = np.floor((ranks - 1) * n_buckets / stocks_per_date) + 1
    return buckets.astype(int)


def bucket_returns(df, score_col, n_buckets=5, label_col="fwd_1m"):
    # bucket n_buckets = highest score
    work = df[["Date", label_col]].copy()
    work["bucket"] = assign_buckets(df, score_col, n_buckets)

    # rows = dates, columns = buckets, values = mean return of the stocks in that bucket
    bucket_means_by_date = work.groupby(["Date", "bucket"])[label_col].mean().unstack()

    # average over all dates
    average_bucket_returns = bucket_means_by_date.mean()
    return bucket_means_by_date, average_bucket_returns


def monotonicity(bucket_means):
    # 1.0 = returns go up with every bucket (perfect staircase), -1.0 = go down
    bucket_numbers = pd.Series(bucket_means.index, index=bucket_means.index)
    return spearman_corr(bucket_numbers, bucket_means)


def get_top_k_rows(df, score_col, k):
    # rank 1 = highest score, ties broken by row order
    ranks = df.groupby("Date")[score_col].rank(method="first", ascending=False)
    top_rows = df[ranks <= k]
    return top_rows


def top_k_returns(df, score_col, k, label_col="fwd_1m"):
    # equal-weight average return of the k highest-score stocks on each date
    top_rows = get_top_k_rows(df, score_col, k)
    monthly_returns = top_rows.groupby("Date")[label_col].mean()
    return monthly_returns


def top_k_turnover(df, score_col, k):
    top_rows = get_top_k_rows(df, score_col, k)

    # the set of top-k tickers on each date
    names_by_date = {}
    for date, one_month in top_rows.groupby("Date"):
        names_by_date[date] = set(one_month["Ticker"])

    dates = sorted(names_by_date.keys())

    # fraction of names that are new compared with last month
    changed_fractions = []
    for i in range(1, len(dates)):
        names_kept = len(names_by_date[dates[i - 1]] & names_by_date[dates[i]])
        changed_fractions.append(1 - names_kept / k)

    return np.mean(changed_fractions)


def performance_stats(monthly_returns, benchmark_returns):
    # only use dates that both series have
    common_dates = monthly_returns.index.intersection(benchmark_returns.index)
    strategy_returns = monthly_returns.loc[common_dates]
    benchmark_on_dates = benchmark_returns.loc[common_dates]

    excess_returns = strategy_returns - benchmark_on_dates
    mean_excess = excess_returns.mean()
    sharpe_excess = excess_returns.mean() / excess_returns.std() * np.sqrt(12)
    hit_rate = (excess_returns > 0).mean() * 100

    # max drawdown of the strategy's own cumulative curve (starting from 1)
    cumulative_curve = (1 + strategy_returns).cumprod()
    running_peak = np.maximum(cumulative_curve.cummax(), 1)
    max_drawdown = (cumulative_curve / running_peak - 1).min()

    stats = {
        "mean_excess": mean_excess,
        "sharpe_excess": sharpe_excess,
        "hit_rate": hit_rate,
        "max_drawdown": max_drawdown,
        "n_months": len(common_dates),
    }
    return stats


def long_short(df, score_col, n_buckets=5, label_col="fwd_1m"):
    # top bucket minus bottom bucket on each date
    bucket_means_by_date, average_bucket_returns = bucket_returns(df, score_col, n_buckets, label_col)
    long_short_returns = bucket_means_by_date[n_buckets] - bucket_means_by_date[1]
    return long_short_returns


def evaluate_score(df, score_col, k_list=[5, 10], benchmark_ew=None, benchmark_nifty=None):
    summary = {}

    # rank ic
    ic_series = rank_ic_by_date(df, score_col)
    summary.update(ic_summary(ic_series))

    # buckets
    bucket_means_by_date, average_bucket_returns = bucket_returns(df, score_col)
    summary["monotonicity"] = monotonicity(average_bucket_returns)
    for bucket_number in average_bucket_returns.index:
        summary["bucket_" + str(bucket_number)] = average_bucket_returns[bucket_number]

    # long short
    long_short_returns = long_short(df, score_col)
    summary["long_short_mean"] = long_short_returns.mean()
    summary["long_short_sharpe"] = long_short_returns.mean() / long_short_returns.std() * np.sqrt(12)

    # if no equal-weight benchmark is given, the mean of all 30 stocks is the same thing
    if benchmark_ew is None:
        benchmark_ew = df.groupby("Date")["fwd_1m"].mean()

    # top-k portfolios
    for k in k_list:
        prefix = "top" + str(k) + "_"
        monthly_returns = top_k_returns(df, score_col, k)
        summary[prefix + "mean_return"] = monthly_returns.mean()

        ew_stats = performance_stats(monthly_returns, benchmark_ew)
        summary[prefix + "excess_ew"] = ew_stats["mean_excess"]
        summary[prefix + "sharpe_ew"] = ew_stats["sharpe_excess"]
        summary[prefix + "hit_rate_ew"] = ew_stats["hit_rate"]
        summary[prefix + "max_drawdown"] = ew_stats["max_drawdown"]

        if benchmark_nifty is not None:
            nifty_stats = performance_stats(monthly_returns, benchmark_nifty)
            summary[prefix + "excess_nifty"] = nifty_stats["mean_excess"]
            summary[prefix + "sharpe_nifty"] = nifty_stats["sharpe_excess"]
            summary[prefix + "hit_rate_nifty"] = nifty_stats["hit_rate"]

        summary[prefix + "turnover"] = top_k_turnover(df, score_col, k)

    return summary


def bootstrap_mean_ci(values, n_boot=2000, seed=42, ci_level=90):
    # resample the months with replacement and look at how much the mean moves around
    values = np.array(values.dropna())
    random_state = np.random.RandomState(seed)

    boot_means = []
    for i in range(n_boot):
        sample = random_state.choice(values, size=len(values), replace=True)
        boot_means.append(sample.mean())

    lower_percentile = (100 - ci_level) / 2
    upper_percentile = 100 - lower_percentile
    lower = np.percentile(boot_means, lower_percentile)
    upper = np.percentile(boot_means, upper_percentile)
    return lower, upper
