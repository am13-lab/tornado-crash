from abc import abstractmethod
from typing import Any, Iterable, List, Optional, Set, Tuple
from dataclasses import is_dataclass

import pandas as pd


class BaseDataReader:
    def __init__(self, schema: Set[str] | Any) -> None:
        assert isinstance(schema, set) or is_dataclass(schema)
        self.schema = schema if isinstance(schema, set) else {*vars(schema()).keys()}
    @abstractmethod
    def _extract(
        self,
        *args,
        **kwargs,
    ) -> pd.DataFrame | List[pd.DataFrame]:
        """
        Return raw deposit, withdraw dataframes in order
        """
        raise NotImplementedError

    @abstractmethod
    def _preprocess(
        self,
        raw_df: pd.DataFrame | List[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Return deposit, withdraw dataframes after preprocessing
        """
        raise NotImplementedError

    def _validate(
        self,
        raw_df: pd.DataFrame | List[pd.DataFrame],
    ):
        """
        Check whethere dataframe structure is as expected
        """
        if isinstance(raw_df, pd.DataFrame):
            assert len(raw_df) > 0, "dataframe cannot be empty"
            assert set(raw_df.columns) == self.schema, "missing expected column(s)"
        elif isinstance(raw_df, list):
            for i in range(len(raw_df)):
                assert len(raw_df[i]) > 0, "dataframe cannot be empty"
                assert (
                    set(raw_df[i].columns) == self.schema
                ), "missing expected column(s)"

    def read(
        self,
        n_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Return deposit, withdraw dataframes after preprocessing
        """
        raw_df = self._extract(n_rows=n_rows)
        self._validate(raw_df)
        return self._preprocess(raw_df)


class CSVDataReader(BaseDataReader):
    """
    Load deposit/withdraw data from given csv paths

    Example
    -----------
    >>> r = CSVDataReader()
    >>> df = r.read(n_rows=5000)
    """

    def __init__(self, path: str | List[str], schema: Set[str] | object) -> None:
        super().__init__(schema)
        self.path = path

    def _extract(
        self,
        n_rows: Optional[int] = None,
    ) -> pd.DataFrame | List[pd.DataFrame]:
        if isinstance(self.path, str):
            return pd.read_csv(self.path, nrows=n_rows)
        elif isinstance(self.path, Iterable):
            dfs: List[pd.DataFrame] = [pd.read_csv(p, nrows=n_rows) for p in self.path]
            return dfs

    def __to_timestamp__(self, ts: str):
        return pd.Timestamp(ts_input=ts, unit="s")

    def _preprocess(self, raw_df: pd.DataFrame | List[pd.DataFrame]) -> pd.DataFrame:
        if isinstance(raw_df, pd.DataFrame):
            raw_df["ts"] = raw_df["ts"].apply(self.__to_timestamp__)
            return raw_df
        elif isinstance(raw_df, list):
            for i in range(len(raw_df)):
                raw_df[i]["ts"] = raw_df[i]["ts"].apply(self.__to_timestamp__)
            return raw_df
