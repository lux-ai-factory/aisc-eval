"""Date handling utilities for A4S evaluation.

This module provides utilities for handling date-based operations, particularly
for creating batches of data based on date ranges and iterating over temporal data.
"""

import pandas as pd
import uuid

from a4s_eval.data_model.evaluation import Dataset
from a4s_eval.service.api_client import get_project, get_project_datashape


def get_date_batches(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    date_round: str,
    window: str,
    freq: str,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Generate a list of date batches based on specified parameters.

    Args:
        start_date (pd.Timestamp): The start date of the overall range
        end_date (pd.Timestamp): The end date of the overall range
        date_round (str): How to round the dates (e.g., 'D' for day, 'H' for hour)
        window (str): The size of each window (e.g., '7D' for 7 days)
        freq (str): The frequency of batch starts (e.g., '1D' for daily)

    Returns:
        list[tuple[pd.Timestamp, pd.Timestamp]]: List of (start, end) timestamp pairs
    """
    # Special case: if start_date equals end_date, create a single batch
    if start_date == end_date:
        # Create a batch that includes all data from that single date
        batch_start = start_date
        batch_end = start_date + pd.Timedelta(
            days=1
        )  # Next day to include all data from start_date
        return [(batch_start, batch_end)]

    # Round dates if specified
    start_date_round = start_date
    end_date_round = end_date
    if date_round:
        start_date_round = start_date.floor(date_round)
        end_date_round = end_date.ceil(date_round)

    # Generate the date ranges at specified frequency
    date_ranges = pd.date_range(start_date_round, end_date_round, freq=freq)
    # Calculate window ends by adding the window size to each start date
    windows = date_ranges.to_series().add(pd.Timedelta(window))

    # Filter out windows that extend beyond the end date
    valid_batches = windows[windows <= end_date_round]

    return list(zip(date_ranges[: len(valid_batches)], valid_batches))


class DateIterator:
    """Iterator for processing dataframes in temporal batches.

    This class provides functionality to iterate over a DataFrame in time-based windows,
    useful for temporal analysis and time-series processing.
    """

    def __init__(
        self,
        date_round: str,
        window: str,
        freq: str,
        df: pd.DataFrame,
        date_feature: str,
    ):
        """Initialize the DateIterator.

        Args:
            date_round (str): How to round the dates (e.g., 'D' for day)
            window (str): The size of each window (e.g., '7D' for 7 days)
            freq (str): The frequency of batch starts (e.g., '1D' for daily)
            df (pd.DataFrame): The DataFrame to iterate over
            date_feature (str): The column name containing dates
        """
        df[date_feature] = pd.to_datetime(df[date_feature])
        self.start_date = df[date_feature].min()
        self.end_date = df[date_feature].max()
        self.date_round = date_round
        self.window = window
        self.freq = freq
        self.batches = get_date_batches(
            self.start_date, self.end_date, date_round, window, freq
        )
        self.index = 0
        self.df = df
        self.date_feature = date_feature

    def __iter__(self) -> "DateIterator":
        """Return the iterator object."""
        return self

    def __next__(self) -> tuple[pd.Timestamp, pd.DataFrame]:
        """Get the next batch of data.

        Returns:
            tuple[pd.Timestamp, pd.DataFrame]: A tuple containing the end timestamp
                and the corresponding data slice.

        Raises:
            StopIteration: When there are no more batches to process.
        """
        if self.index >= len(self.batches):
            raise StopIteration
        start, end = self.batches[self.index]
        self.index += 1
        # Filter DataFrame for current time window
        return end, self.df[
            (self.df[self.date_feature] >= start) & (self.df[self.date_feature] < end)
        ].copy()


class ProjectDateIterator:
    """
    TODO: update docstring
    Iterator for processing project datasets in temporal batches.
    """

    def __init__(self, project_pid: uuid.UUID, date_round: str = "1 D"):
        project = get_project(project_pid)
        self.datashape = get_project_datashape(project_pid)
        self.eval_datashape = None
        self.date_round = date_round
        self.date_feature = self.datashape.date.name
        self.window = project.window_size
        self.freq = project.frequency
        self.df = None
        self.index = 0

    def set_dataset(self, dataset: Dataset):
        """
        Allows setting the DataFrame

        """
        df = dataset.data.copy()
        self.eval_datashape = dataset.shape
        df[self.date_feature] = pd.to_datetime(df[self.date_feature])
        self.start_date = df[self.date_feature].min()
        self.end_date = df[self.date_feature].max()
        self.batches = get_date_batches(
            self.start_date, self.end_date, self.date_round, self.window, self.freq
        )
        self.index = 0
        self.df = df

    def __iter__(self) -> "ProjectDateIterator":
        """Return the iterator object."""
        return self

    def __next__(self) -> tuple[pd.Timestamp, Dataset]:
        """Get the next batch of data.

        Returns:
            tuple[pd.Timestamp, Dataset]: A tuple containing the end timestamp
                and the corresponding dataset slice.

        Raises:
            StopIteration: When there are no more batches to process.
        """
        if self.index >= len(self.batches):
            raise StopIteration
        start, end = self.batches[self.index]
        self.index += 1
        # Filter DataFrame for current time window
        temp_df = self.df[
            (self.df[self.date_feature] >= start) & (self.df[self.date_feature] < end)
        ].copy()

        if temp_df.empty:
            # If the filtered DataFrame is empty, skip to the next batch
            return self.__next__()

        # Create a dummy Dataset object for the batch
        dataset = Dataset(pid=uuid.uuid4(), shape=self.eval_datashape, data=temp_df)
        return end, dataset
