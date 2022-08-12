from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Set
from dataclasses import is_dataclass

import pandas as pd
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from nebula3.data.ResultSet import ResultSet
from nebula3.data.DataObject import Value, ValueWrapper


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

        self.__as_map__ = {
            Value.NVAL: "as_null",
            Value.__EMPTY__: "as_empty",
            Value.BVAL: "as_bool",
            Value.IVAL: "as_int",
            Value.FVAL: "as_double",
            Value.SVAL: "as_string",
            Value.LVAL: "as_list",
            Value.UVAL: "as_set",
            Value.MVAL: "as_map",
            Value.TVAL: "as_time",
            Value.DVAL: "as_date",
            Value.DTVAL: "as_datetime",
            Value.VVAL: "as_vertex",
            Value.EVAL: "as_edge",
            Value.PVAL: "as_path",
            Value.GGVAL: "as_geography",
            Value.DUVAL: "as_duration",
        }

    def __enter__(self):
        return self.conn

    def __exit__(self):
        self.conn.close()

    def _cast(self, val: ValueWrapper) -> Any:
        _type = val._value.getType()
        if _type in self.__as_map__:
            return getattr(val, self.__as_map__[_type], lambda *args, **kwargs: None)()
        elif _type == Value.LVAL:
            return [self._cast(x) for x in val.as_list()]
        elif _type == Value.UVAL:
            return {self._cast(x) for x in val.as_set()}
        elif _type == Value.MVAL:
            return {k: self._cast(v) for k, v in val.as_map().items()}
        raise KeyError("No such _type", _type)

    def _extract(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame | List[pd.DataFrame]:
        with self.conn.session_context(self.username, self.password) as sess:
            result: ResultSet = sess.execute_parameter(
                query, params if params != None else {}
            )
            columns = result.keys()
            d: Dict[str, list] = {}
            for col_num in range(result.col_size()):
                col_name = columns[col_num]
                col_list = result.column_values(col_name)
                d[col_name] = [self._cast(x) for x in col_list]
        return pd.DataFrame.from_dict(d)

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
