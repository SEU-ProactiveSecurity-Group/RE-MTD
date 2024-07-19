from enum import Enum


class DefenceStrategy(Enum):
    PORT_HOPPING = "PORT_HOPPING"
    REPLICA_INCREASE = "REPLICA_INCREASE"
    REPLICA_DECREASE = "REPLICA_DECREASE"
    REPLICA_EXPAND = "REPLICA_EXPAND"
    REPLICA_SHRINK = "REPLICA_SHRINK"
    NO_ACTION = "NO_ACTION"


class AttackerType(Enum):
    SC = "SC"
    SV = "SV"
    MC = "MC"
    MV = "MV"


class DefenderType(Enum):
    PORT = "PORT"
    REPLICA = "REPLICA"
    SCALE = "SCALE"
    ALL = "ALL"


def map_action_to_defence(defender_type: DefenderType):
    defence_num = 0
    defender_strategies = {
        DefenderType.PORT: [DefenceStrategy.PORT_HOPPING, DefenceStrategy.NO_ACTION],
        DefenderType.REPLICA: [
            DefenceStrategy.REPLICA_INCREASE,
            DefenceStrategy.REPLICA_DECREASE,
            DefenceStrategy.NO_ACTION,
        ],
        DefenderType.SCALE: [
            DefenceStrategy.REPLICA_EXPAND,
            DefenceStrategy.REPLICA_SHRINK,
            DefenceStrategy.NO_ACTION,
        ],
        DefenderType.ALL: [
            DefenceStrategy.PORT_HOPPING,
            DefenceStrategy.REPLICA_INCREASE,
            DefenceStrategy.REPLICA_DECREASE,
            DefenceStrategy.REPLICA_EXPAND,
            DefenceStrategy.REPLICA_SHRINK,
            DefenceStrategy.NO_ACTION,
        ],
    }
    defence_map = {}
    defence_num = len(defender_strategies[defender_type])
    for i in range(defence_num):
        defence_map[i] = defender_strategies[defender_type][i]
    return defence_map, defence_num


def check_defender_type(defender_type: str) -> DefenderType:
    if defender_type not in DefenderType.__members__:
        raise ValueError(f"Invalid defence type: {defender_type}")
    return DefenderType.__members__[defender_type]


def check_attacker_type(attacker_type: str) -> AttackerType:
    if attacker_type not in AttackerType.__members__:
        raise ValueError(f"Invalid attacker type: {attacker_type}")
    return AttackerType.__members__[attacker_type]
