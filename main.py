import os
from ioutil.reader import CSVDataReader, NebulaDataReader
from ioutil.writer import to_json
from heuristics import (
    ExactMatchHeuristic,
    GasPriceHeuristic,
    MultipleDenominationHeuristic,
    LinkedTransactionHeuristic,
    Sequential,
)
from models.schema import TornadoTxs
from dotenv import load_dotenv


load_dotenv()

if __name__ == "__main__":
    deposit_path = "./data/tornado_deposit.csv"
    withdraw_path = "./data/tornado_withdraw.csv"
    reader = CSVDataReader(path=[deposit_path, withdraw_path], schema=TornadoTxs)
    deposit_df, withdraw_df = reader.read(n_rows=90818)

    nebula_reader = NebulaDataReader(
        address=[(os.environ["address"], int(os.environ["port"]))],
        username=os.environ["username"],
        password=os.environ["password"],
    )

    hrstcs: Sequential = [
        ExactMatchHeuristic("exact_match"),
        MultipleDenominationHeuristic("same_num_tx"),
        GasPriceHeuristic("gas_price"),
        LinkedTransactionHeuristic("linked_tx", nebula_reader),
    ]

    os.makedirs("proceed", exist_ok=True)

    for hrstc in hrstcs:
        df, addr_sets = hrstc.run(deposit_df, withdraw_df)
        df.to_csv("proceed/{}.csv".format(hrstc._name), index=False)
        to_json(addr_sets, "proceed/{}_related_addr.json".format(hrstc._name))
