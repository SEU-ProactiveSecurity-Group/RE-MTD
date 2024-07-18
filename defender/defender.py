from .all import AllDefender

def defenderFactory(env, defender_type):
    if defender_type == "ALL":
        return AllDefender(env)
    # elif defender_type == "PORT":
    #     return PortDefender()
    # elif defender_type == "REPLICA":
    #     return ReplicaDefender()
    # elif defender_type == "SCALE":
    #     return ScaleDefender()
    return None
