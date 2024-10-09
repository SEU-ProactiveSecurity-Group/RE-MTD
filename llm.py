from tqdm import tqdm
import numpy as np
import random
import time
import os
from openai import OpenAI
from pydantic import BaseModel
from torch.utils.tensorboard import SummaryWriter
from typing import List


class Think(BaseModel):
    action: int
    function: List[str]
    exit: bool


class Action(BaseModel):
    action: int
    reward: float


class LLM:
    def __init__(self, action_dim):
        self.client = OpenAI()
        self.intial_prompt = [
            {
                "role": "system",
                "content": "你是一个世界级智能代理，能够在交互式环境中与攻击者进行博弈，指导防御者采取最佳防御策略，使得每轮博弈中防御收益最大。",
            },
        ]
        self.total_prompts = []
        self.window = 2
        self.episode_window = 11
        self.prompts = []
        self.action_dim = action_dim
        self.rnd = 0

    def reset(self):
        self.total_prompts = []
        self.prompts = []
        self.action = random.randint(0, self.action_dim - 1)
        self.rnd = 0

    def cut_prompts(self):
        if len(self.prompts) > self.window * self.episode_window:
            self.prompts = self.total_prompts[-self.window * self.episode_window :]

    def act(
        self, state, step_defence, calculate_reward, reset_step_defence, train=True
    ):
        self.rnd += 1
        # 理解当前服务状态
        prompts = [
            {
                "role": "user",
                "content": f"第{self.rnd}轮博弈开始时，当前防御方服务状态为 state: {state}",
            },
            {
                "role": "assistant",
                "content": "【问题】 在当前服务状态state，防御方采取哪个防御动作action，使防御收益reward最大",
            },
            {
                "role": "assistant",
                "content": """【思考】我可以枚举从0到5的所有防御动作action，对于每个防御动作action，可以模拟执行防御动作action更新服务状态，再计算防御收益reward，选取其中收益reward最大的防御动作action。
                可以采取的工具函数有：step_defence, calculate_reward, reset_step_defence，你可以调用这些函数来帮助你计算：
                step_defence，防御者执行防御动作action，返回防御后的服务状态，当你想要知道执行防御动作action对应的服务状态时，可以调用该函数；
                calculate_reward，计算防御后的奖励并返回奖励值，当你执行防御动作后想要计算奖励值时，可以调用该函数；
                reset_step_defence，重置防御者服务状态，当你想要重置防御者服务状态时并进入下一轮模拟时，可以调用该函数
                下面开始模拟。""",
            },
        ]

        functions = ["step_defence", "calculate_reward", "reset_step_defence"]
        if train:
            print("train\n")
            for i in range(self.action_dim):
                step_defence(i)
                reward = calculate_reward()
                reset_step_defence()
                prompt = [
                    {
                        "role": "assistant",
                        "content": "【思考】 下一步要采取动作action是什么，调用函数顺序为是什么",
                    },
                    {
                        "role": "assistant",
                        "content": f"当前为第{i}轮模拟，采取action为{i}，调用函数顺序为{functions}，得到的奖励为{reward}，是否结束模拟{i == self.action_dim - 1}",
                    },
                ]
                prompts += prompt
                print(prompt[1]["content"])
        else:
            print("test\n")
            is_end = False
            i = 0
            while not is_end:
                completion = self.client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=self.intial_prompt + self.prompts + prompts,
                    response_format=Think,
                )
                reason = completion.choices[0].message.parsed
                if reason.action != i:
                    prompt = [
                        {
                            "role": "assistant",
                            "content": f"【模拟】 当前为第{i}轮模拟，action应该等于{i}，重新进行本轮模拟",
                        }
                    ]
                    print(prompt[0]["content"])
                    continue
                if (
                    reason.function[0] == "step_defence"
                    and reason.function[1] == "calculate_reward"
                    and reason.function[2] == "reset_step_defence"
                ):
                    step_defence(reason.action)
                    reward = calculate_reward()
                    reset_step_defence()
                    prompt = [
                        {
                            "role": "assistant",
                            "content": "【思考】 下一步要采取动作action是什么，调用函数顺序为是什么",
                        },
                        {
                            "role": "assistant",
                            "content": f"【模拟】 当前为第{i}轮模拟，采取action为{reason.action}，调用函数顺序为{reason.function}，得到的奖励为{reward}，是否结束模拟{reason.exit}",
                        },
                    ]
                    print(prompt[1]["content"])
                    i += 1
                else:
                    prompt = [
                        {
                            "role": "assistant",
                            "content": f"【模拟】 当前为第{i}轮模拟，调用函数顺序应该为{functions}，重新进行本轮模拟",
                        }
                    ]
                    print(prompt[0]["content"])
                    continue
                prompts += prompt
                is_end = reason.exit

        prompts += [
            {
                "role": "assistant",
                "content": "【思考】 根据上面模拟结果，选择reward最大的防御动作action",
            }
        ]
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=self.intial_prompt + self.prompts + prompts,
            response_format=Action,
        )
        action = completion.choices[0].message.parsed.action
        prompt = {
            "role": "assistant",
            "content": f"选择reward最大的防御动作action为{action}",
        }
        print(prompt["content"])
        prompts.append(prompt)

        self.prompts += prompts
        self.total_prompts += prompts

        return action


def train_and_test(env, prefix, num_episodes):
    action_dim = 6
    agent = LLM(action_dim)

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
    txt_r = open(f"./txt/{prefix}-{title}/return.json", "w", encoding="utf-8")
    txt_a = open(f"./txt/{prefix}-{title}/action.json", "w", encoding="utf-8")
    txt_p = open(f"./txt/{prefix}-{title}/prompts.json", "w", encoding="utf-8")
    txt_s = open(f"./txt/{prefix}-{title}/state.json", "w", encoding="utf-8")

    for i in range(num_episodes):
        terminated = False
        elapsed_steps = 0
        max_episode_steps = 1
        return_list = []
        action_list = []
        state_list = []

        state = env.reset()
        txt_s.write(f"{state}\n")
        agent.reset()
        with tqdm(total=max_episode_steps, desc=f"iteration {i}") as pbar:

            def step_defence(action):
                env.step_defence(action)

            def reset_step_defence():
                env.reset_step_defence()

            def calculate_reward():
                reward = env.calculate_reward()
                return reward

            if i in [0, 1]:
                # train
                while not terminated and (elapsed_steps < max_episode_steps):

                    # train
                    # agent.cut_prompts()
                    attack_state = env.step_attack()
                    action = agent.act(
                        attack_state,
                        step_defence,
                        calculate_reward,
                        reset_step_defence,
                        train=True,
                    )
                    step_defence(action)
                    reward = calculate_reward()

                    next_state = env.state.copy()
                    state = next_state
                    elapsed_steps += 1
            else:
                # test
                while not terminated and (elapsed_steps < max_episode_steps):
                    # agent.cut_prompts()
                    attack_state = env.step_attack()
                    action = agent.act(
                        attack_state,
                        step_defence,
                        calculate_reward,
                        reset_step_defence,
                        train=False,
                    )
                    step_defence(action)
                    reward = calculate_reward()

                    next_state = env.state.copy()
                    state = next_state
                    elapsed_steps += 1

                    state_list.append(state)
                    action_list.append(action)
                    return_list.append(reward)

                    writer_r.add_scalar(
                        "return",
                        reward,
                        elapsed_steps,
                    )

                    pbar.set_postfix(
                        {
                            "episode": elapsed_steps,
                            "return": "%.3f" % reward,
                        }
                    )
                    pbar.update(1)
                pbar.total = elapsed_steps
                txt_r.write(f"{return_list}\n")
                txt_a.write(f"{action_list}\n")
                txt_p.write(f"{agent.total_prompts}\n")
                txt_s.write(f"{state}\n")

    txt_r.close()
