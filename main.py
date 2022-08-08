import os
from ioutil.reader import CSVDataReader
from ioutil.writer import to_json
from heuristics import (
    ExactMatchHeuristic,
    GasPriceHeuristic,
    SameNumTransactionHeuristic,
    LinkedTransactionHeuristic,
    Sequential
)
from models.schema import TornadoTxs

if __name__ == "__main__":
    deposit_path = "./data/tornado_deposit.csv"
    withdraw_path = "./data/tornado_withdraw.csv"
    reader = CSVDataReader(path=[deposit_path, withdraw_path], schema=TornadoTxs)
    deposit_df, withdraw_df = reader.read(n_rows=25000)

    hrstcs: Sequential = [
        ExactMatchHeuristic("exact_match"),
        SameNumTransactionHeuristic("same_num_tx"),
        GasPriceHeuristic("gas_price"),
        LinkedTransactionHeuristic("linked_tx"),
    ]

    os.makedirs("proceed", exist_ok=True)

    for hrstc in hrstcs:
        df, addr_sets = hrstc.run(deposit_df, withdraw_df)
        df.to_csv("proceed/{}.csv".format(hrstc._name), index=False)
        to_json(addr_sets, "proceed/{}_related_addr.json".format(hrstc._name))
