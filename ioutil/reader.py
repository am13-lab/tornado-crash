from abc import abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
from dataclasses import is_dataclass

import pandas as pd
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from nebula3.data.ResultSet import ResultSet
from nebula3.common.constants import Row


class BaseDataReader:
    def __init__(self) -> None:
        pass

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
    ) -> pd.DataFrame | List[pd.DataFrame]:
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
    ) -> pd.DataFrame | List[pd.DataFrame]:
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
        assert isinstance(schema, set) or is_dataclass(schema)
        self.schema = schema if isinstance(schema, set) else {*vars(schema()).keys()}
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

    def _preprocess(
        self, raw_df: pd.DataFrame | List[pd.DataFrame]
    ) -> pd.DataFrame | List[pd.DataFrame]:
        if isinstance(raw_df, pd.DataFrame):
            raw_df["ts"] = raw_df["ts"].apply(self.__to_timestamp__)
            return raw_df
        elif isinstance(raw_df, list):
            for i in range(len(raw_df)):
                raw_df[i]["ts"] = raw_df[i]["ts"].apply(self.__to_timestamp__)
            return raw_df


class NebulaDataReader(BaseDataReader):
    def __init__(
        self,
        address: Any,
        username: str,
        password: str,
        config: Optional[Config] = None,
    ) -> None:
        """
        address: [(ip, port)]
        """
        config = Config() if config is None else config
        config.max_connection_pool_size = 10

        self.conn = ConnectionPool()
        assert self.conn.init(address, config)

        self.username = username
        self.password = password

    def __enter__(self):
        return self.conn

    def __exit__(self):
        self.conn.close()

    def _extract(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame | List[pd.DataFrame]:
        with self.conn.session_context(self.username, self.password) as sess:
            result: ResultSet = sess.execute_parameter(
                query, params if params != None else {}
            )
            column_name: List[str] = result.keys()
            arr: Dict[List[Any]] = {}
            for c_cnt in range(result.col_size()):
                tmp: List[Any] = []
                for r_cnt in range(result.row_size()):
                    col = result.row_values(r_cnt)[c_cnt]
                    if col.is_empty():
                        tmp.append(None)
                    elif col.is_null():
                        tmp.append(None)
                    elif col.is_bool():
                        tmp.append(col.as_bool())
                    elif col.is_int():
                        tmp.append(col.as_int())
                    elif col.is_double():
                        tmp.append(col.as_double())
                    elif col.is_string():
                        tmp.append(col.as_string())
                    elif col.is_time():
                        tmp.append(col.as_time())
                    elif col.is_date():
                        tmp.append(col.as_date())
                    elif col.is_datetime():
                        tmp.append(col.as_datetime())
                    elif col.is_list():
                        tmp.append(col.as_list())
                    elif col.is_set():
                        tmp.append(col.as_set())
                    elif col.is_map():
                        tmp.append(col.as_map())
                    elif col.is_vertex():
                        tmp.append(col.as_node())
                    elif col.is_edge():
                        tmp.append(col.as_relationship())
                    elif col.is_path():
                        tmp.append(col.as_path())
                    elif col.is_geography():
                        tmp.append(col.as_geography())
                    else:
                        print("ERROR: Type unsupported")
                        return
                    tmp.append(col)
                arr[column_name[c_cnt]] = tmp
        return pd.DataFrame(arr)

    def _preprocess(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        return raw_df

    def read(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ):
        raw_df = self._extract(query, params)
        self.conn.close()
        return self._preprocess(raw_df)
