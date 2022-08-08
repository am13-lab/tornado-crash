:auto USING PERIODIC COMMIT LOAD CSV  WITH HEADERS FROM "file:///tornado_deposit_txs.csv" AS line
WITH line LIMIT 5000

MERGE (f:address {address: line.from_address})
MERGE (t:address {address: line.to_address})
MERGE (f)-[r:INTERACT_WITH {txhash: line.txhash, value: line.value}]->(t)