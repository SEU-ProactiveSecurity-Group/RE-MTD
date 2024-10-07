from tqdm import tqdm
import numpy as np
import random
import time
import os
from openai import OpenAI
from pydantic import BaseModel
from torch.utils.tensorboard import SummaryWriter


class Action(BaseModel):
    action: int


class LLM:
    def __init__(self, ser_max_num, ser_ind, pod_max_num, pod_con_num, action_dim):
        self.client = OpenAI()
        self.messages = [
            {
                "role": "system",
                "content": "你是一个可以自我学习对防御策略进行迭代的安全机器人，攻击者会通过ldos攻击占用防御者的边缘节点连接资源，而防御者通过三种mtd策略来防御攻击，你需要做的是每轮通过环境状态来决策下一轮中防御者采取的某一个mtd策略，降低攻击者攻击效果。",
            },
            {
                "role": "system",
                "content": f"用户每轮会输入三个变量 state、next_state 和 reward。state 和 next_state 都是{ser_max_num}*{ser_ind}的二维环境状态矩阵，state 表示上一轮防御者状态，next_state 表示攻防博弈结束后这一轮防御者状态。其中第一维表示防御者拥有的边缘节点数，第二维表示防御者所有边缘节点的服务指标，如 state[0][0]、state[0][1]、state[0][2] 分别表示第一个边缘节点的副本数量、服务连接数、端口号信息，其中，副本数量取值范围为 0 到 {pod_max_num}，服务连接数取值范围为 0 到 {pod_con_num}，端口号取值范围为 30000 到 32767。reward 是一个标量，用于评价上一轮防御效果，如果 reward 越大，表示防御效果越好，reward 越小，表示防御效果越差。",
            },
            {
                "role": "system",
                "content": """
                    你需要根据每次输入的 state、next_state 和 reward 信息，推断攻击者下一步攻击意图，并且给出防御动作 action，action 取值范围为 0 到 5，其中每种策略表示含义如下：
                    0 表示采取端口跳变，即通过重新分配节点的端口号，可以中断节点攻击者流量，达到短暂防御效果；
                    1 表示增加副本，即选定服务过载的节点，增加节点的副本数量和原服务一样，并将所有流量的一半给副本，但是收到副本总数限制；
                    2 表示减少副本，即删除单个节点的副本，并将删除的副本的流量平摊到各个其他服务，但是服务的副本数量不能小于 1；
                    3 表示节点扩容，即将负载率高于75%的节点的节点容量扩大，扩容后保证流量负载率为75%，保证服务质量；
                    4 表示节点缩容，即将负载率低于50%的节点容量缩小，缩容后保证流量负载率为75%，保证资源利用率；
                    5 表示不采取任何防御动作，即保持当前状态。
                """,
            },
            {
                "role": "system",
                "content": "总结说来，你的任务是根据每轮输入的 state 和 reward，学习攻击者攻击策略，得到防御者防御策略，输出最佳防御动作 action，使下一轮 reward 值最大。",
            },
        ]
        self.response_format = Action
        self.action = random.randint(0, action_dim - 1)

    def take_action(self):
        return self.action

    def update(self, state, next_state, reward):
        self.messages.append(
            {
                "role": "user",
                "content": f"state: {state}, next_state: {next_state}, reward: {reward}",
            }
        )
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=self.messages,
            response_format=self.response_format,
        )

        # print(completion.choices[0].message.parsed.action)

        self.action = completion.choices[0].message.parsed.action
        return self.action


def train_and_test(env, prefix, num_episodes):
    ser_max_num = env.ser_max_num
    ser_ind = env.ser_ind
    pod_max_num = env.pod_max_num
    pod_con_num = env.pod_con_num
    agent = LLM(ser_max_num, ser_ind, pod_max_num, pod_con_num, 6)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    title = (
        env.defender.type.value
        + "-"
        + env.attacker.type.value
        + "-"
        + str(env.attacker.num)
        + "("
        + timestamp
        + ")"
    )
    writer_r = SummaryWriter(f"./log/{prefix}-{title}/return")
    os.makedirs(f"./txt/{prefix}-{title}", exist_ok=True)
    txt_r = open(f"./txt/{prefix}-{title}/return.txt", "w", encoding="utf-8")
    txt_a = open(f"./txt/{prefix}-{title}/action.txt", "w", encoding="utf-8")

    for i in range(num_episodes):
        with tqdm(total=num_episodes, desc=f"iteration {i}") as pbar:
            terminated = False
            elapsed_steps = 0
            episode_return = 0
            max_episode_steps = 20
            return_list = []
            action_list = []

            state = env.reset()

            while not terminated and (elapsed_steps < max_episode_steps):
                action = agent.take_action()
                next_state, reward, terminated = env.step(action)

                elapsed_steps += 1
                state = next_state
                episode_return += reward
                return_list.append(episode_return)

                agent.update(state, next_state, reward)

                action_list.append(action)
                return_list.append(reward)

                avg_return = np.mean(return_list[-5:])
                writer_r.add_scalar(
                    "return",
                    avg_return,
                    elapsed_steps,
                )
                pbar.set_postfix(
                    {
                        "episode": elapsed_steps,
                        "return": "%.3f" % avg_return,
                    }
                )

                pbar.update(1)

            txt_r.write(f"{return_list}\n")
            txt_a.write(f"{action_list}\n")

    txt_r.close()
