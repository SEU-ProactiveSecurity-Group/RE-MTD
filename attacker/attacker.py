from .sc import SCAttacker
from .sv import SVAttacker
from .mc import MCAttacker
from .mv import MVAttacker
from constants import AttackerType


def attackerFactory(env, attacker_type, attacker_num=10):
    if attacker_type == AttackerType.SC:
        return SCAttacker(env, attacker_num)
    elif attacker_type == AttackerType.SV:
        return SVAttacker(env, attacker_num)
    elif attacker_type == AttackerType.MC:
        return MCAttacker(env, attacker_num)
    elif attacker_type == AttackerType.MV:
        return MVAttacker(env, attacker_num)
    return None
