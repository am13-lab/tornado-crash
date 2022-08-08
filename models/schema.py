from dataclasses import dataclass
from pandas import Timestamp


@dataclass
class TornadoTxs:
    ts: Timestamp = ''
    txhash: str = ''
    address: str = ''
    tornado_cash_address: str = ''
    value: float = ''
    gas_price: float = ''
