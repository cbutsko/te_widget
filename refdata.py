import glob
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import requests
from loguru import logger

MIN_EDGE_BUFFER = 2
NUM_TIMESTEPS = 12

def month_diff(month1: int, month2: int) -> int:
    """This function computes the difference between `month1` and `month2`
    assuming that `month1` is in the past relative to `month2`.
    The difference is calculated such that it falls within the range of 0 to 12 months.

    Parameters
    ----------
    month1 : int
        The reference month (1-12).
    month2 : int
        The month to compare against (1-12).

    Returns
    -------
    int
        The difference between `month1` and `month2`.
    """

    return (month2 - month1) % 12


def get_best_valid_time(
    row: pd.Series, buffer: int, num_timesteps: int
) -> Union[pd.Timestamp, float]:
    """
    Determines the best valid time for a given row of data based on specified shift constraints.

    This function evaluates potential valid times by shifting the original valid time
    forward or backward according to the values in 'valid_month_shift_forward' and
    'valid_month_shift_backward' fields. It ensures the shifted time remains within
    the period defined by 'start_date' and 'end_date' with sufficient buffer.

    Parameters
    ----------
    row : pd.Series
        A pandas Series containing the following fields:
        - valid_time: The original valid time
        - start_date: The start date of the allowed period
        - end_date: The end date of the allowed period
        - valid_month_shift_forward: Number of months to shift forward
        - valid_month_shift_backward: Number of months to shift backward
    buffer : int
        Buffer in months to apply when aligning available extractions with user-defined temporal extent.
        Determines how close we allow the true valid_time of the sample to be to the edge of the processing period.
    num_timesteps : int
        The number of timesteps accepted by the model.
        This is used to define the middle of the user-defined period.

    Returns
    -------
    datetime or np.nan
        The best valid time after applying shifts, or np.nan if no valid time can be found.
        If both forward and backward shifts are valid, the choice depends on the relative
        magnitude of the shifts compared to buffer.
    """

    def is_within_period(proposed_date, start_date, end_date):
        return (proposed_date - pd.DateOffset(months=buffer) >= start_date) & (
            proposed_date + pd.DateOffset(months=buffer) <= end_date
        )

    def check_shift(proposed_date, valid_time, start_date, end_date):
        proposed_start_date = proposed_date - pd.DateOffset(
            months=(num_timesteps // 2 - 1)
        )
        proposed_end_date = proposed_date + pd.DateOffset(months=(num_timesteps // 2))
        return (
            is_within_period(proposed_date, start_date, end_date)
            & (valid_time >= proposed_start_date)
            & (valid_time <= proposed_end_date)
        )

    valid_time = row["valid_time"]
    start_date = row["start_date"]
    end_date = row["end_date"]

    proposed_valid_time_fwd = valid_time + pd.DateOffset(
        months=row["valid_month_shift_forward"]
    )
    proposed_valid_time_bwd = valid_time - pd.DateOffset(
        months=row["valid_month_shift_backward"]
    )

    shift_forward_ok = check_shift(
        proposed_valid_time_fwd, valid_time, start_date, end_date
    )
    shift_backward_ok = check_shift(
        proposed_valid_time_bwd, valid_time, start_date, end_date
    )

    if not shift_forward_ok and not shift_backward_ok:
        return np.nan
    if shift_forward_ok and not shift_backward_ok:
        return proposed_valid_time_fwd
    if not shift_forward_ok and shift_backward_ok:
        return proposed_valid_time_bwd
    if shift_forward_ok and shift_backward_ok:
        return (
            proposed_valid_time_bwd
            if (row["valid_month_shift_backward"] - row["valid_month_shift_forward"])
            <= buffer
            else proposed_valid_time_fwd
        )