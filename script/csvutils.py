from glob import glob
import os
from typing import Iterable, Tuple
from web3 import Web3
import pandas as pd


class EOADetector:
    def __init__(self, provider) -> None:
        self.w3 = Web3(Web3.HTTPProvider(provider))

    def _is_eoa(self, addr) -> bool:
        addr = self.w3.toChecksumAddress(addr)
        return (
            self.w3.eth.get_code(addr, block_identifier=self.w3.eth.default_block).hex()
            == "0x"
        )

    def filter_eoa(
        self, df: pd.DataFrame, col: str, verbose: bool = True
    ) -> pd.DataFrame:
        """
        filter only eoa by specified column
        """
        assert col in df.columns
        df["is_eoa"] = df[col].apply(self._is_eoa)
        if verbose:
            print("=========================")
            print(df["is_eoa"].value_counts())
            print("=========================")
        return df[df["is_eoa"] == True]


def _merge_csv(files: Iterable) -> pd.DataFrame:
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def _build_vertex_and_edge(
    deposit_tx_df: pd.DataFrame,
    withdraw_tx_df: pd.DataFrame,
    tornado_deposit_df: pd.DataFrame,
    tornado_withdraw_df: pd.DataFrame,
):
    """'
    Parameter
    -------
    - both input contains only eoa, however, deposit_tx_df is those `from_address` participate
        in tornado cash deposit event, and withdraw_tx_df is formed via `to_address` in tornado cash withdraw event.

    Returns
    ----------
    - Find all unique address as vertex
    - For each unique address, labeled "deposit, withdraw" as vertexs' property

    """

    df: pd.DataFrame = pd.concat([deposit_tx_df, withdraw_tx_df], ignore_index=True)
    series: pd.Series = pd.concat(
        [df["from_address"], df["to_address"]], ignore_index=True
    )

    # A mapping from address to its unique index
    unique_addr = series.unique()
    addr2idx = {addr: idx for idx, addr in enumerate(unique_addr)}

    deposit_addr_set = set(tornado_deposit_df["address"].unique())
    withdraw_addr_set = set(tornado_withdraw_df["address"].unique())

    vertex = pd.DataFrame.from_dict(
        {
            "vid": [addr2idx[x] for x in unique_addr],
            "address": unique_addr,
            "depositor": [
                True if x in deposit_addr_set else False for x in unique_addr
            ],
            "withdrawer": [
                True if x in withdraw_addr_set else False for x in unique_addr
            ],
        }
    )

    df["from_address"] = df["from_address"].apply(lambda x: addr2idx[x])
    df["to_address"] = df["to_address"].apply(lambda x: addr2idx[x])
    relation_transfer = df[["from_address", "to_address", "txhash", "value"]]

    return vertex, relation_transfer


def save_related_tx_to_graph_csv(dir: str):
    deposit_files = glob(os.path.join(dir, "depositor/*.csv"))
    deposit_df = _merge_csv(deposit_files)

    withdraw_files = glob(os.path.join(dir, "withdrawer/*.csv"))
    withdraw_df = _merge_csv(withdraw_files)

    tornado_deposit_df = pd.read_csv("./data/tornado_deposit.csv")
    tornado_withdraw_df = pd.read_csv("./data/tornado_withdraw.csv")

    vertex, edge = _build_vertex_and_edge(
        deposit_df, withdraw_df, tornado_deposit_df, tornado_withdraw_df
    )
    vertex.to_csv("vertex.csv", index=False, header=False)
    edge.to_csv("relation.csv", index=False, header=False)


if __name__ == "__main__":
    save_related_tx_to_graph_csv("./data/tornado_related_txs")
