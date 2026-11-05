# -*- coding: utf-8 -*-

import numpy as np

__all__ = ["mae", "mse", "rmse", "mape", "smape", "mase", 'wape', 'msmape', "mae_norm", "mse_norm", "rmse_norm", "mape_norm", "smape_norm", "mase_norm", 'wape_norm', 'msmape_norm']


def _error(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """ Simple error """
    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """ Percentage error """
    return (actual - predicted) / actual


def mse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))


def mae(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """ Mean Absolute Error """

    return np.mean(np.abs(_error(actual, predicted)))


def mase(
    actual: np.ndarray,
    predicted: np.ndarray,
    hist_data: np.ndarray,
    seasonality: int = 1,
    **kwargs
):
    """
    Mean Absolute Scaled Error
    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    actual, predicted, history = (np.asarray(x, dtype=np.float64) for x in (actual, predicted, hist_data))
    if actual.shape != predicted.shape or not actual.size or actual.ndim not in (1, 2):
        raise ValueError("MASE requires equally shaped nonempty forecast and target arrays.")
    if history.ndim != actual.ndim or history.shape[1:] != actual.shape[1:]:
        raise ValueError("MASE history must preserve the forecast channel schema.")
    if isinstance(seasonality, bool) or not isinstance(seasonality, (int, np.integer)) or not 0 < seasonality < len(history):
        raise ValueError("MASE seasonality must be a positive integer shorter than the training history.")
    differences = np.abs(history[seasonality:] - history[:-seasonality])
    observed = np.isfinite(differences)
    counts = observed.sum(axis=0)
    scale = np.divide(np.where(observed, differences, 0).sum(axis=0), counts,
                      out=np.full_like(counts, np.nan, dtype=np.float64), where=counts > 0)
    scale = np.where(scale > 0, scale, np.nan)
    return np.mean(np.abs(actual - predicted) / scale)


def mape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """
    Mean Absolute Percentage Error
    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0
    """
    return np.mean(np.abs(_percentage_error(actual, predicted))) * 100


def smape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """
    Symmetric Mean Absolute Percentage Error
    """
    return (
        np.mean(
            2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)))
        )
        * 100
    )

def wape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """Masked weighted absolute percentage error (WAPE)

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
    Returns:
        torch.Tensor: masked mean absolute error
    """
    loss = np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100
    return loss

def msmape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 0.1, **kwargs):
    """
    Function to calculate series wise smape values

    Parameters
    forecasts - a matrix containing forecasts for a set of series
                no: of rows should be equal to number of series and no: of columns should be equal to the forecast horizon
    test_set - a matrix with the same dimensions as 'forecasts' containing the actual values corresponding with them
    """

    denom = np.maximum(0.5 + epsilon, np.abs(predicted) + np.abs(actual) + epsilon)
    msmape_per_series = np.mean(2 * np.abs(predicted - actual) / denom) * 100
    return msmape_per_series


def _error_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object,  **kwargs):
    """ Simple error """
    return scaler.transform(actual) - scaler.transform(predicted)


def _percentage_error_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """ Percentage error """
    return (scaler.transform(actual) - scaler.transform(predicted)) / scaler.transform(actual)


def mse_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """ Mean Squared Error """
    return np.mean(np.square(_error_norm(actual, predicted, scaler)))


def rmse_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """ Root Mean Squared Error """
    return np.sqrt(mse_norm(actual, predicted, scaler))


def mae_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """ Mean Absolute Error """

    return np.mean(np.abs(_error_norm(actual, predicted, scaler)))


def mase_norm(
    actual: np.ndarray,
    predicted: np.ndarray,
    scaler: object,
    hist_data: np.ndarray,
    seasonality: int = 1,
    **kwargs
):
    """
    Mean Absolute Scaled Error
    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    actual = scaler.transform(actual)
    predicted = scaler.transform(predicted)
    hist_data = scaler.transform(hist_data)
    return mase(actual, predicted, hist_data, seasonality=seasonality)


def mape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """
    Mean Absolute Percentage Error
    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0
    """
    return np.mean(np.abs(_percentage_error_norm(actual, predicted, scaler))) * 100


def smape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """
    Symmetric Mean Absolute Percentage Error
    """
    actual = scaler.transform(actual)
    predicted = scaler.transform(predicted)
    return (
        np.mean(
            2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)))
        )
        * 100
    )

def wape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    """Masked weighted absolute percentage error (WAPE)

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
    Returns:
        torch.Tensor: masked mean absolute error
    """
    actual = scaler.transform(actual)
    predicted = scaler.transform(predicted)
    loss = np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100
    return loss

def msmape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, epsilon: float = 0.1, **kwargs):
    """
    Function to calculate series wise smape values

    Parameters
    forecasts - a matrix containing forecasts for a set of series
                no: of rows should be equal to number of series and no: of columns should be equal to the forecast horizon
    test_set - a matrix with the same dimensions as 'forecasts' containing the actual values corresponding with them
    """
    actual = scaler.transform(actual)
    predicted = scaler.transform(predicted)
    denom = np.maximum(0.5 + epsilon, np.abs(predicted) + np.abs(actual) + epsilon)
    msmape_per_series = np.mean(2 * np.abs(predicted - actual) / denom) * 100
    return msmape_per_series
