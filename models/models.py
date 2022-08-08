from enum import Enum


class Entity(Enum):
    EOA = 0
    DEPOSIT = 1
    EXCHANGE = 2
    DEX = 3
    DEFI = 4
    ICO_WALLETS = 5
    MINING = 6
    TORNADO = 7


class Heuristics(Enum):
    DEPO_REUSE = 0
    SAME_ADDR = 1
    GAS_PRICE = 2
    SAME_NUM_TX = 3
    LINKED_TX = 4
    TORN_MINE = 5
    DIFF2VEC = 6

