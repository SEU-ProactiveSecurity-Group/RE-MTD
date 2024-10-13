from tqdm import tqdm
import numpy as np
from tenacity import retry, stop_after_attempt
import time
import os
from openai import OpenAI
from pydantic import BaseModel
from torch.utils.tensorboard import SummaryWriter


class Action(BaseModel):
    action: int
    desc: str


class Judge(BaseModel):
    success: bool
    desc: str


class Reflex(BaseModel):
    desc: str


class LLM:
    def __init__(self):
        self.client = OpenAI()
        self.inital_prompts = [
            {
                "role": "system",
                "content": "你是一个可以在多轮ldos攻击中不断改进防御策略的安全机器人，每一轮攻防中，攻击者会先通过ldos攻击占用防御者的服务连接资源，然后防御者选择六种mtd策略之一来防御攻击，再通过防御结果是成功还是失败来反思优化防御策略。由此可以将一回合的防御分为下面三个阶段：决策、判断、反思。",
            },
            {
                "role": "system",
                "content": """
                在决策阶段，你需要根据给出的当前服务状态 state，如果服务处于不正常状态，则需要选择一种防御动作 action 将服务恢复至正常，如果处于正常状态，则需要尽量减少资源消耗，具体评价指标后面介绍。
                服务状态 state 是一个 10*3 的二维数组，表示最多有 10 个服务副本，每个副本有 3 个状态信息可以被监测，分别是副本的资源节点数量、连接数和端口号：
                    副本资源节点数量 state[0][:] 是一个 0 到 100 之间的整数，如果节点数量不为 0，则表示该服务副本存在，否则不是一个服务副本；
                    副本连接数 state[1][:] 为副本的所有节点数的连接数总和，节点连接数在 0 到 256 之间，一个副本最多有 100 个节点，所以副本连接数取值在 0 到 25600 之间；
                    副本端口号 state[2][:] 取值在 30000 到 32767 之间。
                防御动作 action 是一个 0 到 5 之间的整数，每个整数对应一种 mtd 防御策略：
                    0 表示采取端口跳变，即通过重新分配副本的端口号，可以中断攻击者流量，达到短暂防御效果，但是会导致副本时延增加；
                    1 表示增加副本，即选定过载服务副本，创建一份拷贝副本，并将其所有流量的一半给拷贝副本，但是受到副本总数限制；
                    2 表示减少副本，即删除某个副本，并将删除的副本的流量平摊到各个其他副本上，提高资源利用率，但是副本数量不能小于 1；
                    3 表示副本扩容，即通过增加副本的节点数，将负载率高于75%的副本的容量扩大，扩容后保证流量负载率为75%，但是受到资源节点数量限制；
                    4 表示副本缩容，即通过减少副本的节点数，将负载率低于50%的节点容量缩小，缩容后保证流量负载率为75%，提高资源利用率；
                    5 表示不采取任何防御动作，即保持当前状态。
                """,
            },
            {
                "role": "system",
                "content": """
                在判断阶段，防御者执行完防御动作并得到此时服务状态的评价指标 R_s、R_e、R_d、R_b，你需要根据此指标来判断防御是否成功，输出成功或失败 success 和失败原因 desc：
                评价指标的定义如下：
                    R_s 表示正常服务率，即 0.5 <= 副本连接数 / (副本节点数 * 节点最大连接数) <= 0.75 的服务副本占副本总数的比例，R_s = 正常服务副本数 / 总副本数；
                    R_e 表示低效服务率，即 副本连接数 / (副本节点数 * 节点最大连接数) < 0.5 的服务副本占副本总数的比例，R_e = 低效服务副本数 / 总副本数；
                    R_d 表示危险服务率，即 副本连接数 / (副本节点数 * 节点最大连接数) > 0.75 的服务副本占副本总数的比例，R_d = 危险服务副本数 / 总副本数；
                    R_b 表示服务中断次数，即端口变换次数。
                防御失败条件为：
                    R_s < 0.8 或 R_e > 0.5 或 R_d > 0.2 或 R_b > 2。
                """,
            },
        ]
        self.prompts = []
        self.memory_size = 1
        self.memory_prompts = []

    def reset(self):
        if len(self.prompts) >= 8:
            self.memory_prompts.append(self.prompts)
            self.memory_prompts = self.memory_prompts[-self.memory_size :]
        self.prompts = []

    @retry(stop=stop_after_attempt(3))
    def remember(self):
        print("remember")
        # r = np.random.randint(0, 1)
        r = 1
        prompts = [
            {
                "role": "assistant",
                "content": "【回忆】 这是前几轮防御成功的经验，你可以根据这些经验来优化本轮防御动作。" if r == 0 else "【回忆】 这是前几轮防御成功经验，你需要采取和与之不同的防御动作，探索更好的防御策略，并且每一步尽量不要采取相同动作！",
            },
            {
                "role": "assistant",
                "content": f"{self.memory_prompts}",
            },
        ]
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=self.inital_prompts + prompts,
            response_format=Reflex,
            timeout=30,
        )
        parsed = completion.choices[0].message.parsed
        print(parsed.desc)

    @retry(stop=stop_after_attempt(3))
    def take_action(self, state, step):
        print("action")
        prompts = [
            {
                "role": "user",
                "content": f"第{step}轮攻防博弈开始时，当前防御方服务状态为 state: {state}",
            },
            {
                "role": "assistant",
                "content": "【决策】 在当前服务状态 state，防御方采取哪个防御动作 action 及原因 desc，能够成功防御本轮攻击，并且使正常服务率 R_s 最大，低效服务率 R_e 和危险服务率 R_d 尽可能小？",
            },
        ]
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=self.inital_prompts + self.prompts + prompts,
            response_format=Action,
            timeout=30,
        )
        parsed = completion.choices[0].message.parsed
        print(parsed.action)
        prompts += [
            {
                "role": "assistant",
                "content": f"本轮采取的防御动作为 {parsed.action}，原因为 {parsed.desc}",
            }
        ]
        self.prompts += prompts
        return parsed.action

    @retry(stop=stop_after_attempt(3))
    def judge(self, R_s, R_e, R_d, R_b):
        print("judge")
        prompts = [
            {
                "role": "user",
                "content": f"防御结果判断，得到的评价指标为：R_s={R_s}，R_e={R_e}，R_d={R_d}，R_b={R_b}",
            },
            {
                "role": "assistant",
                "content": "【判断】 防御成功或失败，如果失败原因是什么？",
            },
        ]
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=self.inital_prompts + self.prompts + prompts,
            response_format=Judge,
            timeout=30,
        )
        parsed = completion.choices[0].message.parsed
        prompts += [
            {
                "role": "assistant",
                "content": f"本轮防御{'失败，失败原因为：' + parsed.desc if not parsed.success else '成功'}",
            }
        ]
        self.prompts += prompts
        return parsed.success, parsed.desc

    @retry(stop=stop_after_attempt(3))
    def reflex(self, defence_state):
        print("reflex")
        prompts = [
            {
                "role": "user",
                "content": f"防御结束时进行反思，开始时服务状态为 state，执行防御动作 action 后服务状态变为 defence_state: {defence_state}，并且得到评价指标 R_s，R_e，R_d，R_b 以及防御结果。",
            },
            {
                "role": "assistant",
                "content": "【反思】 通过本轮防御，如果成果请说明是否还有优化地方，如果失败请反思更好的防御动作。",
            },
        ]
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=self.inital_prompts + self.prompts + prompts,
            response_format=Reflex,
            timeout=30,
        )
        parsed = completion.choices[0].message.parsed
        prompts += [
            {
                "role": "assistant",
                "content": f"本轮防御经验为：{parsed.desc}",
            }
        ]
        self.prompts += prompts


def train_and_test(env, prefix, num_episodes):
    agent = LLM()

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
    writer_success = SummaryWriter(f"./log/{prefix}-{title}/success")
    writer_s = SummaryWriter(f"./log/{prefix}-{title}/s")
    writer_e = SummaryWriter(f"./log/{prefix}-{title}/e")
    writer_d = SummaryWriter(f"./log/{prefix}-{title}/d")
    writer_b = SummaryWriter(f"./log/{prefix}-{title}/b")
    os.makedirs(f"./txt/{prefix}-{title}", exist_ok=True)
    txt_r = open(f"./txt/{prefix}-{title}/return.json", "w", encoding="utf-8")
    txt_a = open(f"./txt/{prefix}-{title}/action.json", "w", encoding="utf-8")
    txt_p = open(f"./txt/{prefix}-{title}/prompt.json", "w", encoding="utf-8")

    for i in range(num_episodes):
        success = True
        step = 0
        max_steps = 15
        episode_return = []
        return_list = []
        action_list = []
        state = env.reset()
        agent.reset()
        agent.remember()
        with tqdm(total=max_steps, desc=f"iteration {i}") as pbar:
            while success and (step < max_steps):
                action = agent.take_action(state, step)
                next_state, defence_state, R_s, R_e, R_d, R_b = env.step(action)
                success, fail_msg = agent.judge(R_s, R_e, R_d, R_b)
                agent.reflex(defence_state)

                episode_return.append([success, fail_msg, R_s, R_e, R_d, R_b])
                step += 1
                state = next_state

                return_list.append(episode_return)
                action_list.append(action)

                pbar.set_postfix(
                    {
                        "episode": step,
                        "return": "%.3f" % R_s,
                    }
                )

                pbar.update(1)

            sucess_num = 0
            for item in episode_return:
                if item[0]:
                    sucess_num += 1

            s_sum = 0
            e_sum = 0
            d_sum = 0
            b_sum = 0
            for item in episode_return[-5:]:
                s_sum += item[2]
                e_sum += item[3]
                d_sum += item[4]
                b_sum += item[5]
            s = s_sum / 5
            e = e_sum / 5
            d = d_sum / 5
            b = b_sum / 5

            writer_success.add_scalar(
                "success",
                sucess_num,
                i,
            )
            writer_s.add_scalar(
                "s",
                s,
                i,
            )
            writer_e.add_scalar(
                "e",
                e,
                i,
            )
            writer_d.add_scalar(
                "d",
                d,
                i,
            )
            writer_b.add_scalar(
                "b",
                b,
                i,
            )

            txt_r.write(f"{return_list}\n")
            txt_a.write(f"{action_list}\n")
            txt_p.write(f"{agent.prompts}\n")

    txt_r.close()
