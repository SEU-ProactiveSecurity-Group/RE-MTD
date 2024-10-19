import argparse
from argparse import Namespace
from env import Env
# from dqn import train_and_test
from llm import train_and_test
from constants import check_defender_type, check_attacker_type

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="argparse")

    # Add arguments
    parser.add_argument(
        "--prefix", type=str, required=False, help="Prefix of the log file"
    )
    parser.add_argument(
        "--attacker_type", type=str, required=True, help="Type of the attacker"
    )
    parser.add_argument(
        "--attacker_num", type=int, required=True, help="Number of attackers"
    )
    parser.add_argument(
        "--defender_type", type=str, required=True, help="Type of the defender"
    )

    # 根据命令行参数选择环境
    args = parser.parse_args()
    
    # 校验参数
    defender_type = check_defender_type(args.defender_type)
    attacker_type = check_attacker_type(args.attacker_type)
    env_args = Namespace(
        attacker_type=attacker_type,
        defender_type=defender_type,
        attacker_num=args.attacker_num,
    )

    env = Env(env_args)
    prefix = args.prefix if args.prefix else "default"
    train_and_test(env, prefix, 5)