from tqdm import tqdm
import numpy as np
from tenacity import retry, stop_after_attempt
import time
import os
import json
from openai import OpenAI
from pydantic import BaseModel
from torch.utils.tensorboard import SummaryWriter


class Action(BaseModel):
    action: int
    con_percent: float
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
                "content": """你是一个可以在多回合ldos攻击中不断改进防御策略的安全机器人，每一回合中有多轮攻防过程。
                每轮开始时需要判断服务是否处于危险状态，如果处于正常状态（所有副本连接负载率小于0.6）则不采取动作或者回收副本节点等（例如动作 2, 4, 5），保证资源利用率；如果服务处于危险状态（某些副本连接负载量过载），则采取动作（例如动作 0, 1, 3）防御攻击者。
                在每一轮攻防中，如果有攻击者流量，攻击者会先通过ldos攻击占用防御者的服务连接资源，然后防御者选择六种mtd策略之一来防御攻击，再通过判断防御结果是成功还是失败来优化下一轮防御动作，即一轮中先后进行两个阶段：决策、判断。
                如果防御失败或者成功防御 max_step_num 轮，则本回合攻防结束，并且可以得到本轮防御成功或失败。
                在每回合开始前，会先进入 反思 阶段，即回顾上回合所有轮次攻防，总结成功或失败经验，用于改进本一回合防御策略。
                """,
            },
            {
                "role": "system",
                "content": """
                在决策阶段，你需要根据给出的当前服务状态 state，需要初步判断服务是否处于危险状态，然后输出一个防御动作 action，以及连接数负载率阈值 con_percent，来防御攻击者流量或减少资源消耗，保证服务正常运行。
                服务状态 state 是一个 10*3 的二维数组，表示最多有 10 个服务副本，每个副本有 3 个状态信息可以被监测，分别是副本的资源节点数量、连接数和端口号：
                    副本资源节点数量 state[0][:] 是一个 0 到 100 之间的整数，如果节点数量不为 0，则表示该服务副本存在，否则不是一个服务副本；
                    副本连接数 state[1][:] 为副本的所有节点数的连接数总和，节点连接数在 0 到 256 之间，一个副本最多有 100 个节点，所以副本连接数取值在 0 到 25600 之间；
                    副本端口号 state[2][:] 取值在 30000 到 32767 之间。
                防御动作 action 是一个 0 到 5 之间的整数，每个整数对应一种 mtd 防御策略，有些动作带有 con_percent 参数，表示连接数负载率阈值，
                其中，连接数负载率 = 副本连接数 / (副本节点数 * 节点最大连接数)，action 详细定义如下：
                    0 表示采取端口跳变，即通过重新分配副本的端口号，可以清除所有副本的攻击者流量，但是受到服务中断次数限制，在防御方资源不足（例如因为副本数量达到最大或者没有剩余资源节点供分配时）情况下考虑使用！
                    1 表示增加副本，即选定连接负载率超过 con_percent 的所有服务副本，创建一份拷贝副本，并将其所有流量的一半给拷贝副本，但是受到副本总数限制；
                    2 表示减少副本，即删除连接负载率小于 con_percent 的所有服务副本，并将删除的副本的流量平摊到各个其他副本上，提高资源利用率，但是副本数量不能小于 1；
                    3 表示副本扩容，即通过增加副本的节点数，将连接负载率高于 con_percent 的所有副本容量扩大，扩容后保证流量负载率为 con_percent，但是受到资源节点数量限制；
                    4 表示副本缩容，即通过减少副本的节点数，将连接负载率低于 con_percent 的所有副本容量缩小，缩容后保证流量负载率为 con_percent，提高资源利用率；
                    5 表示不采取任何防御动作，即保持当前状态。
                """,
            },
            {
                "role": "system",
                "content": """
                在判断阶段，防御者首先执行动作，得到动作执行成功或失败，防御者资源不够用等情况可能导致动作执行失败，如果执行成功则说明给出的 action 和 con_percent 是有效的，如果执行失败则可能下一轮需要选择执行其他动作。
                执行完防御动作后则可以得到此时服务状态的评价指标 R_s、R_e、R_d、R_b，你需要根据此指标来判断防御是否成功，输出成功或失败 success 和失败原因 desc：
                评价指标的定义如下：
                    R_d 表示危险服务率，即 连接数负载率 > 0.9 的服务副本占副本总数的比例，R_d = 危险服务副本数 / 总副本数；
                    R_e 表示低效服务率，即 连接数负载率 < 0.3 的服务副本占副本总数的比例，R_e = 低效服务副本数 / 总副本数；
                    R_b 表示服务中断次数，即端口变换次数。
                    R_c 表示服务时延，即 服务总连接数 / 服务最大连接数，公式展开来即 所有副本的 (副本节点数 * 节点连接数) 之和 / 所有副本的 (副本节点数 * 节点最大连接数) 之和。
                防御失败条件为：
                   R_d > 0 或 R_c > 0.8 或 R_b > 4。
                资源利用评价：
                    在保证防御成功前提下，要求 R_e 尽可能小。
                服务质量评价：
                    在保证防御成功前提下，要求 R_b 尽可能小、R_c 尽可能小。
                """,
            },
            {
                "role": "system",
                "content": """
                在反思阶段，首先输入上回合是否防御成功，如果失败则反思失败原因和失败动作序列，然后判断上回合轮失败动作是否与之前回合失败动作重复，如果重复则警告本轮不要重复失败动作！！
                另外给出之前所有回合的失败动作序列和成功动作序列，对于失败动作序列，需要避免重复失败动作；对于成功动作序列，需要参考成功动作序列指导本回合防御决策！！
                最后输出反思结果 desc。
                """,
            },
        ]
        self.prompts = []
        self.actions = []
        self.fail_actions = []
        self.success_actions = []

    def reset(self):
        self.prompts = []

    @retry(stop=stop_after_attempt(3))
    def take_action(self, state, step):
        print("action")
        prompts = [
            {
                "role": "user",
                "content": f"第{step}轮攻防博弈开始时，当前防御方服务状态为 state: {str(state.tolist())}",
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
        print(parsed.action, parsed.con_percent)
        self.actions.append([parsed.action, parsed.con_percent])
        prompts += [
            {
                "role": "assistant",
                "content": f"本轮准备采取的防御动作为 {parsed.action}，连接负载率阈值为 {parsed.con_percent}，原因为 {parsed.desc}",
            }
        ]
        self.prompts += prompts
        return parsed.action, parsed.con_percent

    @retry(stop=stop_after_attempt(3))
    def judge(
        self, defence_state, defence_success, defence_fail_msg, R_d, R_e, R_b, R_c
    ):
        print("judge")
        prompts = [
            {
                "role": "user",
                "content": f"本次执行防御动作 action {'成功' if defence_success else ('失败，原因为' + defence_fail_msg)}，之后服务状态变为 defence_state: {str(defence_state.tolist())}，并且得到的评价指标为：R_e={R_e}，R_d={R_d}，R_b={R_b}。",
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

    def judge_fail(
        self, defence_state, defence_success, defence_fail_msg, R_d, R_e, R_b, R_c
    ):
        print("judge_fail")
        success, fail_msg = False, ""
        if R_d > 0:
            success = False
            fail_msg = "R_d > 0，有副本处于危险状态"
        elif R_c > 0.8:
            success = False
            fail_msg = "R_c > 0.8，服务的时延过高"
        elif R_b > 4:
            success = False
            fail_msg = "R_b > 4，服务中断次数过多"
        else:
            success = True
            fail_msg = None
        prompts = [
            {
                "role": "user",
                "content": f"本次执行防御动作 action {'成功' if defence_success else ('失败，原因为' + defence_fail_msg) + '，下回合可能需要采取其他动作。'}，之后服务状态变为 defence_state: {str(defence_state.tolist())}，并且得到的评价指标为：R_d={R_d}，R_e={R_e}，R_b={R_b}，R_c={R_c}。",
            },
            {
                "role": "assistant",
                "content": "【判断】 防御成功或失败，如果失败原因是什么？",
            },
        ]
        prompts += [
            {
                "role": "assistant",
                "content": f"本轮防御{'失败，失败原因为：' + fail_msg if not success else '成功'}",
            }
        ]

        self.prompts += prompts
        return success, fail_msg

    @retry(stop=stop_after_attempt(3))
    def reflex(self, step_num, success, fail_msg):
        print("reflex")

        repeated_fail_actions = True
        repeated_success_actions = True
        if len(self.fail_actions) == 0:
            repeated_fail_actions = False
        if len(self.success_actions) == 0:
            repeated_success_actions = False
        if not success:
            for a in self.fail_actions:
                if a["actions"] != self.actions:
                    repeated_fail_actions = False
                    break
            if not repeated_fail_actions:
                self.fail_actions.append(
                    {"actions": self.actions, "fail_reason": fail_msg}
                )
        else:
            for a in self.success_actions:
                if a != self.actions:
                    repeated_success_actions = False
                    break
            if not repeated_success_actions:
                self.success_actions.append(self.actions.copy())

        prompts = [
            {
                "role": "user",
                "content": f"上回合攻防结束，经过了{step_num}轮攻防，防御{'成功' if success else ('失败，失败原因为：' + fail_msg + '。失败动作序列为：' + str(self.actions))}。",
            },
            {
                "role": "user",
                "content": f"""上回合 { '重复了前面回合的失败动作' if repeated_fail_actions else '没有重复前面回合失败动作' }。到目前为止所有回合防御失败的动作列表为 fail_actions：{str(self.fail_actions)}，请避免本回合防御动作与之重复！！到目前为止所有回合防御成功的动作列表为 success_actions：{str(self.success_actions)}，请参考成功动作序列指导本回合防御决策。"""
            },
            {
                "role": "assistant",
                "content": "【反思】 通过对上回合攻防过程进行反思，如果成功则总结成功经验指导本回合防御决策，如果失败则反思失败原因并在本回合采取不同的防御策略。不要纠结于每轮的防御胜利，需要规划整个回合里面的每轮的防御动作才能确保一回合的胜利！",
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
                "content": f"上回合防御经验为：{parsed.desc}",
            }
        ]
        self.actions = []
        self.prompts += prompts


def train_and_test(env, prefix, num_episodes, attack_sequence):
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
    writer_e = SummaryWriter(f"./log/{prefix}-{title}/e")
    writer_d = SummaryWriter(f"./log/{prefix}-{title}/d")
    writer_b = SummaryWriter(f"./log/{prefix}-{title}/b")
    writer_c = SummaryWriter(f"./log/{prefix}-{title}/c")
    os.makedirs(f"./txt/{prefix}-{title}", exist_ok=True)
    r_arr = []
    a_arr = []
    p_arr = []
    txt_r = open(f"./txt/{prefix}-{title}/return.json", "w", encoding="utf-8")
    txt_a = open(f"./txt/{prefix}-{title}/action.json", "w", encoding="utf-8")
    txt_p = open(f"./txt/{prefix}-{title}/prompt.json", "w", encoding="utf-8")

    for_step = 0
    for_episode_success = False

    for i in range(num_episodes):
        success = True
        step = 0
        fail_msg = ""
        max_steps = len(attack_sequence)
        episode_return = []
        return_list = []
        action_list = []
        state = env.reset()
        agent.reset()

        if i != 0:
            agent.reflex(for_step, for_episode_success, for_episode_fail_msg)

        with tqdm(total=max_steps, desc=f"iteration {i}") as pbar:
            while success and (step < max_steps):
                do_attack = attack_sequence[step]
                action, con_percent = agent.take_action(state, step)
                (
                    next_state,
                    defence_state,
                    defence_success,
                    defence_fail_msg,
                    R_e,
                    R_d,
                    R_b,
                    R_c,
                ) = env.step(action, {"con_percent": con_percent}, do_attack)
                # success, fail_msg = agent.judge(R_s, R_e, R_d, R_b)
                success, fail_msg = agent.judge_fail(
                    defence_state, defence_success, defence_fail_msg, R_d, R_e, R_b, R_c
                )

                episode_return.append(
                    [
                        success,
                        fail_msg,
                        defence_success,
                        defence_fail_msg,
                        R_d,
                        R_e,
                        R_b,
                        R_c,
                    ]
                )
                step += 1
                state = next_state

                return_list.append(episode_return)
                action_list.append([action, con_percent])

                pbar.set_postfix(
                    {
                        "episode": step,
                        "return": "%.3f" % (success / max_steps),
                    }
                )

                pbar.update(1)

        print(
            f"第{i}回合结束，总共进行了{step}轮攻防，本回合防御{'成功' if step == max_steps else '失败'}。"
        )

        for_step = step
        for_episode_success = step == max_steps
        for_episode_fail_msg = fail_msg

        sucess_num = 0
        for item in episode_return:
            if item[0]:
                sucess_num += 1

        e_sum = 0
        d_sum = 0
        b_sum = 0
        c_sum = 0
        for item in episode_return[-5:]:
            e_sum += item[4]
            d_sum += item[5]
            b_sum += item[6]
            c_sum += item[7]
        e = e_sum / 5
        d = d_sum / 5
        b = b_sum / 5
        c = c_sum / 5

        writer_success.add_scalar(
            "success",
            sucess_num,
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
        writer_c.add_scalar(
            "c",
            c,
            i,
        )

        r_arr.append(return_list)
        a_arr.append(action_list)
        p_arr.append(agent.prompts)

    json.dump(r_arr, txt_r, ensure_ascii=False, indent=2)
    json.dump(a_arr, txt_a, ensure_ascii=False, indent=2)
    json.dump(p_arr, txt_p, ensure_ascii=False, indent=2)

    writer_success.close()
    writer_b.close()
    writer_d.close()
    writer_e.close()
    writer_c.close()

    txt_r.close()
    txt_a.close()
    txt_p.close()
