from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

WEEK = 7
LAGS = (7, 14, 28)


def load_sales(path: Path) -> pd.DataFrame:
    """Загружает продажи по дням, включая дни без продаж."""
    sales = pd.read_csv(path, usecols=["date", "units", "is_event"])
    sales["date"] = pd.to_datetime(sales["date"], format="%Y-%m-%d", errors="raise")
    sales = sales.sort_values("date", ignore_index=True)
    if sales.empty or sales["date"].isna().any():
        raise ValueError("В каждой строке должна быть дата.")
    if not sales["date"].diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
        raise ValueError("На каждый календарный день должна приходиться одна строка.")
    sales["units"] = pd.to_numeric(sales["units"], errors="raise")
    if not np.isfinite(sales["units"]).all() or sales["units"].lt(0).any():
        raise ValueError("Продажи за день должны быть конечным неотрицательным числом.")
    if not sales["is_event"].isin([0, 1]).all():
        raise ValueError("В is_event допустимы только 0 и 1.")
    return sales


def backtest(sales: pd.DataFrame, n_weeks: int = 12) -> pd.DataFrame:
    """Прогнозирует последние недели по очереди и пополняет историю после каждой недели."""
    first_cutoff = len(sales) - n_weeks * WEEK
    if n_weeks < 1 or first_cutoff < max(LAGS) + WEEK:
        raise ValueError("Нужна хотя бы одна неделя для проверки и достаточно данных для лагов.")

    results = []
    for cutoff in range(first_cutoff, len(sales), WEEK):
        history = sales.iloc[:cutoff]
        week = sales.iloc[cutoff : cutoff + WEEK]
        calendar = week[["date", "is_event"]]
        baseline, catboost = _forecast_week(history, calendar)
        results.append(
            week.rename(columns={"units": "actual"}).assign(
                week_start=week["date"].iloc[0],
                seasonal_naive=baseline,
                catboost=catboost,
            )
        )
    return pd.concat(results, ignore_index=True)


def _forecast_week(
    history: pd.DataFrame, calendar: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    # Для следующих семи дней известен только календарь. Продажи ещё неизвестны.
    dates = pd.concat([history, calendar], ignore_index=True)
    features = pd.DataFrame(
        {"weekday": dates["date"].dt.dayofweek, "is_event": dates["is_event"]}
    )
    for lag in LAGS:
        features[f"lag_{lag}"] = dates["units"].shift(lag)

    train = features.iloc[max(LAGS) : len(history)]
    future = features.iloc[len(history) :]
    model = CatBoostRegressor(
        loss_function="MAE",
        iterations=350,
        depth=4,
        learning_rate=0.05,
        random_seed=42,
        thread_count=2,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train, history["units"].iloc[max(LAGS) :])
    return future["lag_7"].to_numpy(), np.maximum(model.predict(future), 0)
