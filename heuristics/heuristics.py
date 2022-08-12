from collections import defaultdict
import itertools
from pandas import Timedelta, Timestamp
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple
from itertools import combinations
import pandas as pd
import networkx as nx
from tqdm import tqdm

from ioutil.reader import NebulaDataReader

tqdm.pandas()


class BaseHeuristic:
    def __init__(
        self,
        name: str,
    ) -> None:
        self._name: str = name  # name of heuristic process

    @abstractmethod
    def apply_heuristic(
        self,
        deposit_df: pd.DataFrame,
        withdraw_df: pd.DataFrame,
    ) -> Tuple[List[Set[str]], Dict[str, str]]:
        """
        Returns
        ---------
        - clusters: [{txhash1, txhash2, txhash3...},...]
            - a list of clusters which contains related txhashes
        - tx2addrs: {txhash: addr}
            - txhash to depositor/withdrawer
        """
        raise NotImplementedError

    def get_txs(
        self,
        clusters: List[Set[str]],
    ) -> Tuple[Set[str], Dict[str, int]]:
        """
        Returns
        -----------
        - txs: {txhash1, txhash2, txhash3 ...}
            - include all unique txhashes

        - tx2clusterID: {txhash: cluster_id}
            - a mapping from txhash to the cluster_id which txhash belongs to.
        """
        txs: Set[str] = set()
        tx2clusterID: Dict[str, int] = {}
        for idx, cluster in enumerate(clusters):
            txs = txs.union(cluster)
            for tx in cluster:
                tx2clusterID[tx] = idx
        return txs, tx2clusterID

    def get_addr_sets(
        self,
        clusters: List[Set[str]],
        tx2addr: Dict[str, str],
    ) -> List[Set[str]]:
        """
        Store pairs of address that are related to each other.

        Returns
        -----------
        - addr_sets: [{addr1, addr2}, {addr3, addr4}, ...]
            - list of pairs of related addresses
        """
        addr_sets: List[Set[str]] = []
        for cluster in clusters:
            addr_set: List[str] = [*{tx2addr[tx] for tx in cluster}]
            if len(addr_set) > 1:  # make sure not singleton
                addr_sets += [{*x} for x in combinations(addr_set, 2)]
        return addr_sets

    def run(
        self,
        deposit_df: pd.DataFrame,
        withdraw_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[Set[str]]]:
        """
        Main function to execute the heuristics

        Returns
        ---------
        - df: dataframe, contains three columns `tx, addr, cluster`
            - txhash
                - the txhash

            - addr
                - the addr of depositor/withdrawer who involves  in this tx

            - clusterID
                - the ID of cluster which tx belongs to

        - addr_sets: pairs of addrs related to each others
            - will be empty when running ExactMatchHeuristic

        """
        clusters, tx2addr = self.apply_heuristic(deposit_df, withdraw_df)
        txs, tx2clusterID = self.get_txs(clusters)
        addr_sets: List[Set[str]] = self.get_addr_sets(clusters, tx2addr)

        df = pd.DataFrame.from_dict(
            {
                "txhash": [*txs],
                "addr": [tx2addr[tx] for tx in txs],
                "clusterID": [tx2clusterID[tx] for tx in txs],
            }
        )
        return df, addr_sets


class ExactMatchHeuristic(BaseHeuristic):
    """
    By follow conditions, we said two addresses are linked.

    If a number N of deposits with a same address A1, and a number M (M < N) of withdraws
    with same address A1 are detected, then a number M-N of deposit transactions
    must be removed from the anonimity set of all the other withdraw transactions.
    """

    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)

    def __exact_match_heuristic__(
        self,
        deposit_df: pd.DataFrame,
        w_row: pd.DataFrame,
    ) -> Tuple[bool, Optional[List[pd.Series]]]:
        matches: pd.DataFrame = deposit_df[
            (deposit_df.address == w_row.address)
            & (deposit_df.ts < w_row.ts)
            & (deposit_df.tornado_cash_address == w_row.tornado_cash_address)
        ]
        return [matches.iloc[i] for i in range(len(matches))]

    def apply_heuristic(
        self,
        deposit_df: pd.DataFrame,
        withdraw_df: pd.DataFrame,
    ) -> Tuple[List[Set[str]], Dict[str, str]]:
        """
        Iterate over withdraw transactions and apply exact match heuristic
        For each withdraw with matching deposit transactions,
        """
        tx2addr: Dict[str, str] = {}
        graph: nx.DiGraph = nx.DiGraph()

        print("[{}] Iterate over withdraw rows".format(self._name))
        with tqdm(total=len(withdraw_df)) as pbar:
            for w_row in withdraw_df.itertuples():
                deposit_rows: List[pd.Series] = self.__exact_match_heuristic__(
                    deposit_df, w_row
                )
                if len(deposit_rows) > 0:
                    for d_row in deposit_rows:
                        graph.add_nodes_from((w_row.txhash, d_row.txhash))
                        graph.add_edge(w_row.txhash, d_row.txhash)

                        tx2addr[w_row.txhash] = w_row.address
                        tx2addr[d_row.txhash] = d_row.address
                pbar.update()
        clusters: List[Set[str]] = [
            wcc for wcc in nx.weakly_connected_components(graph) if len(wcc) > 1
        ]
        return clusters, tx2addr


class GasPriceHeuristic(BaseHeuristic):
    """
    If there is a deposit and a withdraw transaction with unique gas
    prices (e.g., 3.1415926 Gwei), then we consider the deposit and
    the withdraw transactions linked. The corresponding deposit transaction
    can be removed from any other withdraw transaction’s anonymity set.
    """

    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)

    def __filter_by_unique_gas_price__(self, tx_df: pd.DataFrame) -> pd.DataFrame:
        """ """
        gas_price_count: pd.DataFrame = tx_df[
            ["gas_price", "tornado_cash_address"]
        ].value_counts()
        unique_gas_prices: pd.DataFrame = pd.DataFrame(
            gas_price_count[gas_price_count == 1]
        )
        # tuple set with the values (gas_price, tornado_cash_address) is made
        # to filter efficiently
        tuple_set: Set[Any] = set(
            [(row.Index[0], row.Index[1]) for row in unique_gas_prices.itertuples()]
        )

        output_df: pd.DataFrame = pd.DataFrame(
            filter(
                lambda iter_tuple: (
                    iter_tuple.gas_price,
                    iter_tuple.tornado_cash_address,
                )
                in tuple_set,
                tx_df.itertuples(),
            )
        )
        return output_df

    def __same_gas_price_heuristic__(
        self,
        deposit_df: pd.DataFrame,
        w_row: pd.DataFrame,
    ) -> pd.Series:
        matches: pd.DataFrame = deposit_df[
            (deposit_df.gas_price == w_row.gas_price)
            & (deposit_df.ts <= w_row.ts)
            & (deposit_df.tornado_cash_address == w_row.tornado_cash_address)
        ]

        return matches.iloc[0] if len(matches) > 0 else []

    def apply_heuristic(
        self, deposit_df: pd.DataFrame, withdraw_df: pd.DataFrame
    ) -> Tuple[List[Set[str]], Dict[str, str]]:
        unique_gas_deposit_df = self.__filter_by_unique_gas_price__(deposit_df)

        tx2addr: Dict[str, str] = {}
        graph: nx.DiGraph = nx.DiGraph()

        print("[{}] Iterate over withdraw rows".format(self._name))
        with tqdm(total=len(withdraw_df)) as pbar:
            for _, w_row in withdraw_df.iterrows():
                # apply heuristic for the given withdraw transaction.
                d_row: pd.Series = self.__same_gas_price_heuristic__(
                    unique_gas_deposit_df, w_row
                )

                # when a deposit transaction matching the withdraw transaction
                # gas price is found, add the linked transactions to the dictionary.
                if len(d_row) > 0:

                    graph.add_nodes_from((w_row.txhash, d_row.txhash))
                    graph.add_edge(w_row.txhash, d_row.txhash)

                    tx2addr[w_row.txhash] = w_row.address
                    tx2addr[d_row.txhash] = d_row.address

                pbar.update()

        clusters: List[Set[str]] = [  # ignore singletons
            c for c in nx.weakly_connected_components(graph) if len(c) > 1
        ]
        return clusters, tx2addr


class MultipleDenominationHeuristic(BaseHeuristic):
    """
    If there are multiple (say 12) deposit transactions coming from a deposit
    address and later (within 24 hour) there are 12 withdraw transactions to the same withdraw
    address, then we can link all these deposit transactions to the withdraw
    transactions.
    """

    def __init__(
        self,
        name: str,
        tolerence_hour: int = 24,
    ) -> None:
        super().__init__(name)
        self._tolerence_hour = tolerence_hour

    def __make_portfolio_df__(
        self,
        raw_portfolios: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Params
        --------
        raw_portfolios: {0.1: 2, 1.0: 1}
        """
        deposit_portfolios: Dict[str, List[str]] = defaultdict(lambda: [])
        for porfolio in raw_portfolios:
            for k in [0.10, 1.00, 10.00, 100.00]:
                if k in porfolio:
                    deposit_portfolios[k].append(porfolio[k])
                else:
                    deposit_portfolios[k].append(0)
        deposit_portfolios: Dict[str, List[str]] = dict(deposit_portfolios)
        return pd.DataFrame.from_dict(deposit_portfolios)

    def __get_num_of_withdraws(
        self,
        withdraw_tx: pd.Series,
        withdraw_df: pd.DataFrame,
        time_window: Timestamp,
    ) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
        """
        Given a particular withdraw transaction and the withdraw transactions
        DataFrame, gets the total withdraws the address made in each pool. It
        is returned as a dictionary with the pools as the keys and the number
        of withdraws as the values.
        """
        subset_df: pd.DataFrame = withdraw_df[
            # ignore txs made by others
            (withdraw_df.address == withdraw_tx.address)
            &
            # ignore future transactions
            (withdraw_df.ts <= withdraw_tx.ts)
            &
            # ignore other withdraw transactions not within the last MAX_TIME_DIFF
            (withdraw_df.ts >= (withdraw_tx.ts - time_window))
            &
            # ignore the query row
            (withdraw_df.txhash != withdraw_tx.txhash)
        ]

        withdraw_count: pd.DataFrame = subset_df.groupby("value").size()
        withdraw_count: Dict[str, int] = withdraw_count.to_dict()

        withdraw_txs: pd.DataFrame = subset_df.groupby("value")["txhash"].apply(list)
        withdraw_txs: Dict[str, List[str]] = withdraw_txs.to_dict()

        # add 1 for current address
        if withdraw_tx.value in withdraw_count:
            withdraw_count[withdraw_tx.value] += 1
            withdraw_txs[withdraw_tx.value].append(withdraw_tx.txhash)
        else:
            withdraw_count[withdraw_tx.value] = 1
            withdraw_txs[withdraw_tx.value] = [withdraw_tx.txhash]

        return withdraw_count, withdraw_txs

    def __get_same_num_of_deposits(
        self,
        withdraw_counts: pd.DataFrame,
        deposit_windows: pd.DataFrame,
        deposit_portfolios: pd.DataFrame,
    ) -> List[pd.DataFrame]:
        # simple assertion that the number of non-zero currencies must be the same
        mask: Optional[pd.DataFrame] = (deposit_portfolios > 0).sum(axis=1) == len(
            withdraw_counts
        )
        for k, v in withdraw_counts.items():
            if mask is None:
                mask: pd.DataFrame = deposit_portfolios[k] == v
            else:
                mask: pd.DataFrame = mask & (deposit_portfolios[k] == v)
        return [x[0] for x in deposit_windows[mask].values]

    def __same_num_of_txs_heuristic__(
        self,
        withdraw_tx: pd.Series,
        withdraw_df: pd.DataFrame,
        deposit_windows: pd.DataFrame,
        deposit_portfolios: pd.DataFrame,
        time_window: Timestamp,
    ) -> Dict[str, Any]:
        withdraw_counts, withdraw_set = self.__get_num_of_withdraws(
            withdraw_tx, withdraw_df, time_window
        )

        # remove entries that only give to one pool, we are taking
        # multi-denominational deposits only
        # TODO maybe hackers only save into 100 ETH pool ?
        if len(withdraw_counts) == 1:
            return {}

        # if individual made less than 5 transactions, ignore
        if sum(withdraw_counts.values()) < 5:
            return {}

        withdraw_addr: str = withdraw_tx.address
        withdraw_txs: List[str] = list(itertools.chain(*list(withdraw_set.values())))
        withdraw_tx2addr = dict(
            zip(withdraw_txs, [withdraw_addr for _ in range(len(withdraw_txs))])
        )

        matched_deposits: List[pd.DataFrame] = self.__get_same_num_of_deposits(
            withdraw_counts, deposit_windows, deposit_portfolios
        )

        if len(matched_deposits) == 0:
            return {}

        deposit_addrs: List[str] = []
        deposit_txs: List[str] = []
        deposit_tx2addr: Dict[str, str] = {}

        for match in matched_deposits:
            deposit_addrs.append(match.address.iloc[0])
            txs: List[str] = match.txhash.to_list()
            deposit_txs.extend(txs)
            deposit_tx2addr.update(dict(zip(match.txhash, match.address)))

        deposit_addrs: List[str] = list(set(deposit_addrs))

        privacy_score: float = 1.0 - 1.0 / len(matched_deposits)
        response_dict: Dict[str, Any] = dict(
            withdraw_txs=withdraw_txs,
            deposit_txs=deposit_txs,
            withdraw_addr=withdraw_addr,
            deposit_addrs=deposit_addrs,
            withdraw_tx2addr=withdraw_tx2addr,
            deposit_tx2addr=deposit_tx2addr,
            privacy_score=privacy_score,
        )
        return response_dict

    def apply_heuristic(
        self, deposit_df: pd.DataFrame, withdraw_df: pd.DataFrame
    ) -> Tuple[List[Set[str]], Dict[str, str]]:

        clusters: List[Set[str]] = []
        tx2addr: Dict[str, str] = {}

        time_window: Timestamp = Timedelta(self._tolerence_hour, unit="hours")
        print("[{}] Precomputing deposit windows".format(self._name))
        deposit_windows: pd.Series = deposit_df.progress_apply(
            lambda x: deposit_df[
                # Find all deposits earlier than current one
                (deposit_df.ts <= x.ts)
                &
                # Find all deposits within `time_window` before current one
                (deposit_df.ts >= (x.ts - time_window))
                &
                # Only consider same address
                (deposit_df.address == x.address)
                &
                # Ignore current one from returned set
                (deposit_df.txhash != x.txhash)
            ],
            axis=1,
        )
        deposit_windows: pd.DataFrame = pd.DataFrame(deposit_windows)
        raw_portfolios: pd.DataFrame = deposit_windows.apply(
            lambda x: x.iloc[0].groupby("value").count()["txhash"].to_dict(), axis=1
        )
        print("[{}] Making portfolio".format(self._name))
        deposit_portfolios: pd.DataFrame = self.__make_portfolio_df__(raw_portfolios)

        print("[{}] Iterate over withdraw rows".format(self._name))
        with tqdm(total=len(withdraw_df)) as pbar:
            for w_row in withdraw_df.itertuples():
                response_dict = self.__same_num_of_txs_heuristic__(
                    w_row, withdraw_df, deposit_windows, deposit_portfolios, time_window
                )

                if len(response_dict) > 0:

                    # populate graph with known transactions
                    withdraw_txs: List[str] = response_dict["withdraw_txs"]
                    deposit_txs: List[str] = response_dict["deposit_txs"]
                    withdraw_tx2addr: Dict[str, str] = response_dict["withdraw_tx2addr"]
                    deposit_tx2addr: Dict[str, str] = response_dict["deposit_tx2addr"]
                    tx_cluster: Set[str] = set(withdraw_txs + deposit_txs)

                    tx2addr.update(withdraw_tx2addr)
                    tx2addr.update(deposit_tx2addr)
                    clusters.append(tx_cluster)
                pbar.update()
        return clusters, tx2addr


class LinkedTransactionHeuristic(BaseHeuristic):
    """
    The main goal of this heuristic is to link Ethereum accounts which interacted
    with TCash by inspecting Ethereum transactions outside it.
    This is done constructing two sets, one corresponding to the unique TCash
    deposit addresses and one to the unique TCash withdraw addresses, to
    then make a query to reveal transactions between addresses of each set.
    When a transaction between two of them is found, TCash deposit transactions
    done by the deposit address are linked to all the TCash withdraw transactions
    done by the withdraw address. These two sets of linked transactions are
    filtered, leaving only the ones that make sense. For example, if a deposit
    address A is linked to a withdraw address B, but A made a deposit to the 1
    Eth pool and B made a withdraw to the 10 Eth pool, then this link is not
    considered. Moreover, when considering a particular link between deposit
    and withdraw transactions, deposits done posterior to the latest withdraw are
    removed from the deposit set.
    """

    def __init__(self, name: str, nebula: NebulaDataReader) -> None:
        super().__init__(name)
        self.reader = nebula

    def __linked_tx_heuristic__(
        self,
        deposit_df: pd.DataFrame,
        w_row: pd.DataFrame,
        ext_dict: Dict[str, Set[str]],
    ):
        """
        1. deposit time is earlier than withdraw time
        2. deposit pool is same as withdraw pool
        3. deposit address is related to withdraw address outside Tornado Cash
        """
        if w_row.address not in ext_dict:
            return []
        matches: pd.DataFrame = deposit_df[
            (deposit_df.ts < w_row.ts)
            & (deposit_df.tornado_cash_address == w_row.tornado_cash_address)
            & [
                related_deposit in ext_dict[w_row.address]
                for related_deposit in deposit_df.address
            ]
        ]
        return [matches.iloc[i] for i in range(len(matches))]

    def apply_heuristic(
        self, deposit_df: pd.DataFrame, withdraw_df: pd.DataFrame
    ) -> Tuple[List[Set[str]], Dict[str, str]]:
        print("[{}] Fetching query result from nebula graph".format(self._name))
        ext_df: pd.DataFrame = self.reader.read(
            """
                USE Tornado;
                MATCH 
                    (v1:eoa{withdrawer:True})-[*..1]-(v2:eoa{depositor:True}) 
                WHERE 
                    v1.eoa.address != v2.eoa.address
                RETURN 
                    DISTINCT v1.eoa.address AS v1, toSet(collect(v2.eoa.address)) AS v2
            """,
        )

        # Create a mapping from address to related clusters
        ext_dict: Dict[str, Set[str]] = dict(zip(ext_df["v1"], ext_df["v2"]))

        tx2addr: Dict[str, str] = {}
        graph: nx.DiGraph = nx.DiGraph()

        print("[{}] Iterate over withdraw rows".format(self._name))
        with tqdm(total=len(withdraw_df)) as pbar:
            for w_row in withdraw_df.itertuples():
                pbar.update()
                deposit_rows: List[pd.Series] = self.__linked_tx_heuristic__(
                    deposit_df,
                    w_row,
                    ext_dict,
                )
                if len(deposit_rows) > 0:
                    tx2addr[w_row.txhash] = w_row.address
                    for d_row in deposit_rows:
                        graph.add_nodes_from([w_row.txhash, d_row.txhash])
                        graph.add_edge(w_row.txhash, d_row.txhash)
                        tx2addr[d_row.txhash] = d_row.address
        clusters: List[Set[str]] = [
            wcc for wcc in nx.weakly_connected_components(graph) if len(wcc) > 1
        ]
        return clusters, tx2addr
