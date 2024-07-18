import random
import math
import csv
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os


class ReplayBuffer:
    """经验回放池"""

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, terminated):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, terminated = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), terminated

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


"""
网络结构以ReLU作为激活函数。
"""


class Qnet(torch.nn.Module):
    """只有一层隐藏层的Q网络"""

    def __init__(self, state_dim, hidden1_dim, hidden2_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden1_dim)
        self.fc2 = torch.nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = torch.nn.Linear(hidden2_dim, action_dim)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        x2 = F.relu(self.fc2(x1))
        return self.fc3(x2)


class DQN:
    """DQN算法"""

    def __init__(
        self,
        state_dim,
        hidden1_dim,
        hidden2_dim,
        action_dim,
        learning_rate,
        gamma,
        epsilon,
        target_update,
        device,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden1_dim, hidden2_dim, self.action_dim).to(
            device
        )  # Q网络
        # 目标网络
        self.target_q_net = Qnet(
            state_dim, hidden1_dim, hidden2_dim, self.action_dim
        ).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), betas=(0.9, 0.999), lr=learning_rate
        )

        # 使用learning rate decay
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=40000, gamma=0.2
        )

        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.sample_count = (
            0  # 按一定概率随机选动作，即 e-greedy 策略，并且epsilon逐渐衰减
        )
        self.epsilon_start = 0.1
        self.epsilon_end = 0.001
        self.epsilon_decay = 10000
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        # epsilon衰减
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * math.exp(-1.0 * self.sample_count / self.epsilon_decay)
        # if self.epsilon > self.epsilon_end:  # 随机性衰减
        #     self.epsilon *= self.epsilon_decay
        #     self.epsilon = max(self.epsilon_end, self.epsilon)xaxa

        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            action = self.q_net.forward(state).argmax().item()

        return action

    def predict(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        action = self.q_net.forward(state).argmax().item()

        return action

    def update(self, transition_dict):
        states = (
            torch.tensor(transition_dict["states"], dtype=torch.float)
            .view(-1, 30)
            .to(self.device)
        )
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        next_states = (
            torch.tensor(transition_dict["next_states"], dtype=torch.float)
            .view(-1, 30)
            .to(self.device)
        )
        terminateds = (
            torch.tensor(transition_dict["terminateds"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )

        q_values = self.q_net.forward(states).gather(
            1, actions
        )  # Q值,gather函数用来选取state中的值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net.forward(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (
            1 - terminateds
        )  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数

        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()
        self.scheduler.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络

        count = self.count
        self.count += 1

        return dqn_loss, count


def train_and_test(env, rnd):
    k = 0
    for k in range(rnd):
        random.seed()

        string = str(k + 1)
        title = (
            env.defender.type
            + "-"
            + env.attacker.type
            + "-"
            + str(env.attacker.num)
            + "("
            + string
            + ")"
        )
        writer_l = SummaryWriter("./log/log_{}/loss".format(title))
        writer_r = SummaryWriter("./log/log_{}/return".format(title))

        lr = 1e-3  # 学习速率，Q值迭代时用到
        num_episodes = 10000  # 游戏回合
        max_episode_steps = 50  # 每个episode最大的steps,代表了攻防的总回合
        hidden1_dim = 1024  # 隐藏层的神经单元
        hidden2_dim = 128  # 隐藏层的神经单元
        gamma = 0.98  # 未来回报的折扣率
        epsilon = 0.1  # 贪心搜索
        target_update = 500  # 网络目标更新频率
        buffer_size = 100000  # 存储历史数据的大小
        minimal_size = 5000  # buffer数据的阈值，大于便可以训练
        batch_size = 64  # 每次训练数据的大小
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        replay_buffer = ReplayBuffer(buffer_size)  # 初始化buffer
        state_dim = (
            env.observation_space.shape[0] * env.observation_space.shape[1]
        )  # 状态空间维度
        action_dim = env.action_space.n  # 动作空间维度
        agent = DQN(
            state_dim,
            hidden1_dim,
            hidden2_dim,
            action_dim,
            lr,
            gamma,
            epsilon,
            target_update,
            device,
        )  # dqn算法参数初始化

        csv_dir = "./csv"
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        filename = f"{csv_dir}/{title}.csv"
        csvfile = open(filename, mode="w", newline="")
        fieldnames = ["episode", "avg_return"]
        write = csv.DictWriter(csvfile, fieldnames=fieldnames)
        write.writeheader()

        return_list = []
        for i in range(10):
            with tqdm(total=int(num_episodes / 10), desc="Iteration %d" % i) as pbar:
                for i_episode in range(int(num_episodes / 10)):
                    action_list = []
                    episode_return = 0
                    elapsed_steps = 0
                    state, _ = env.reset()  # 初始化环境状态
                    terminated = False
                    truncated = False
                    while not terminated and not truncated:
                        action = agent.take_action(state.reshape((1, state_dim)))
                        next_state, reward, terminated, truncated, info = env.step(
                            action
                        )
                        elapsed_steps += 1
                        if elapsed_steps >= max_episode_steps:
                            truncated = True
                        # print(terminated, truncated) # 检验游戏进程
                        replay_buffer.add(
                            state.reshape((1, state_dim)),
                            action,
                            reward,
                            next_state.reshape((1, state_dim)),
                            terminated,
                        )
                        state = next_state
                        episode_return += reward  # 回报 = +奖励
                        return_list.append(episode_return)
                        action_list.append(action)  # 打印每轮游戏的action
                        # 当buffer数据的数量超过一定值后,才进行Q网络训练
                        if replay_buffer.size() > minimal_size:
                            b_s, b_a, b_r, b_ns, b_t = replay_buffer.sample(batch_size)
                            transition_dict = {
                                "states": b_s,
                                "actions": b_a,
                                "next_states": b_ns,
                                "rewards": b_r,
                                "terminateds": b_t,
                            }
                            dqn_loss, count = agent.update(transition_dict)

                            writer_l.add_scalar(
                                "training/loss", dqn_loss, count
                            )  # 使用tensotboard画图

                    test_episode = 10
                    if (
                        i_episode
                    ) % 10 == 0:  # 每10轮游戏，进行测试。测试10轮游戏的平均值
                        test_return_list = []
                        for j in range(test_episode):
                            test_state, _ = env.reset()
                            test_episode_return = 0
                            test_elapsed_steps = 0
                            test_terminated = False
                            test_truncated = False
                            test_action_list = []
                            while not test_terminated and not test_truncated:
                                test_action = agent.predict(
                                    test_state.reshape((1, state_dim))
                                )
                                test_action_list.append(test_action)
                                (
                                    test_next_state,
                                    test_reward,
                                    test_terminated,
                                    test_truncated,
                                    test_info,
                                ) = env.step(test_action)
                                test_elapsed_steps += 1
                                if test_elapsed_steps >= max_episode_steps:
                                    test_truncated = True
                                test_state = test_next_state
                                test_episode_return += test_reward  # 回报 = +奖励
                            test_return_list.append(test_episode_return)
                            # print(test_action_list)
                        avg_test_return = np.mean(test_return_list)
                        write.writerow(
                            {
                                "episode": num_episodes / 10 * i + i_episode,
                                "avg_return": avg_test_return,
                            }
                        )
                        writer_r.add_scalar(
                            "training/return",
                            avg_test_return,
                            num_episodes / 10 * i + i_episode,
                        )

                    if (i_episode + 1) % 10 == 0:  # 每10轮游戏，计算平均回报
                        pbar.set_postfix(
                            {
                                "episode": "%d"
                                % (
                                    num_episodes / 10 * i + i_episode + 1
                                ),  # 代表总的轮次
                                "return": "%.3f" % np.mean(return_list[-10:]),
                            }
                        )
                    pbar.update(1)
        csvfile.close()