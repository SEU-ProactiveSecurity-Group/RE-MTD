import argparse
from env import Env
from dqn import train_and_test

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="argparse")

    # Add arguments
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
    env = Env(args)
    train_and_test(env, 1)