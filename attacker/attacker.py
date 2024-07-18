from .sc import SCAttacker
from .sv import SVAttacker
from .mc import MCAttacker
from .mv import MVAttacker


def attackerFactory(env, attacker_type, attacker_num=10):
    if attacker_type == "SC":
        return SCAttacker(env, attacker_num)
    elif attacker_type == "SV":
        return SVAttacker(env, attacker_num)
    elif attacker_type == "MC":
        return MCAttacker(env, attacker_num)
    elif attacker_type == "MV":
        return MVAttacker(env, attacker_num)
    return None
