import gymnasium as gym
from gymnasium import spaces
import numpy as np
from attacker.attacker import attackerFactory
from defender.defender import Defender
from constants import map_action_to_defence


class Env(gym.Env):
    def __init__(self, args):
        self.pod_max_num = 100  # 边缘节点总的计算资源
        self.pod_con_num = 256  # 单个pod最大连接数
        self.ser_max_num = 10  # 最大副本数量
        self.ser_ind = 3  # 服务副本的子指标数量
        self.ser_num = 0  # 当前服务的数量

        self.con_thresh_percent = 0.9  # 正常服务连接数量占比阈值
        self.effective_con_thresh_percent = 0.3  # 高效服务连接数量占比阈值
        self.alpha, self.beta, self.gamma, self.delta = 8, 1, 0.02, 0.5  # 奖励计算权重

        high = np.zeros((self.ser_max_num, self.ser_ind), dtype=np.int64)
        low = np.zeros((self.ser_max_num, self.ser_ind), dtype=np.int64)
        for i in range(self.ser_max_num):
            high[i] = [100, 25600, 32767]
            low[i] = [0, 0, 30000]

        self.observation_space = spaces.Box(
            low, high, shape=(self.ser_max_num, self.ser_ind), dtype=np.int64
        )  # Box（10，3）

        defence_map, defence_num = map_action_to_defence(args.defender_type)
        self.defence_map = defence_map
        self.defence_num = defence_num
        self.action_space = spaces.Discrete(defence_num)  # 动作空间的大小，一维

        self.attacker = attackerFactory(self, args.attacker_type, args.attacker_num)
        self.defender = Defender(self, args.defender_type)
        self.defence_strategy = None

    def reset(self):
        # self.state = None  # 状态矩阵
        # self.for_state = None  # 前一时刻的状态矩阵
        # 服务状态指标：服务的副本数量，服务连接数，服务端口号
        self.state = np.zeros((self.ser_max_num, self.ser_ind), dtype=np.int64)
        self.ser_num = 5
        self.steps_beyond_terminated = 0
        self.port_num = 0  # 端口变换的次数

        self.attack_state = np.zeros(
            (self.ser_max_num, 5), dtype=np.int64
        )  # 攻击者可以观测环境中的信息矩阵:服务端口号，负载率（时延），被攻击次数，攻击权重，攻击流量

        self.defender.reset()
        self.attacker.reset()

        return np.array(self.state, dtype=np.int64)

    def step(self, action, params):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        self.pod_remain = self.pod_max_num - np.sum(
            self.state[:, 0]
        )  # 剩余pod的数量即计算资源
        self.noaction_pen = -1  # 执行动作01234，但是没有采取实质行动的惩罚
        self.port_list = []  # 记录攻击者攻击后，进行端口变换的服务的原来的port
        self.add_ser_list1 = []  # 扩展副本的服务
        self.add_ser_list2 = []  # 扩展副本产生的新服务
        self.del_ser_list = []  # 被删除副本的服务

        # 将 action 转换为对应的 defence_strategy
        defence_strategy = self.defence_map[action]
        defence_success, defence_fail_msg = self.defender.step(defence_strategy, params)  # 执行防御策略
        
        defence_state = self.state.copy()  # 保存前一时刻的状态
        
        self.attacker.step(defence_strategy)  # 根据防御策略执行攻击策略


        # reward奖励函数
        # 第四版：将这一时刻状态与前一时刻对比，得到收益
        no_effective_flag = 0  # 低效服务的数量
        danger_flag = 0  # 危险服务的数量
        pod_num = 0  # 服务的pod数量
        pod_con_num = 0  # 服务的连接数
        for i in range(self.ser_max_num):
            if defence_state[i][0] > 0:
                pod_num += defence_state[i][0]
                pod_con_num += defence_state[i][1]
                if (
                    defence_state[i][1]
                    < self.effective_con_thresh_percent
                    * defence_state[i][0]
                    * self.pod_con_num
                ):
                    no_effective_flag += 1
                elif (
                    defence_state[i][1]
                    > defence_state[i][0] * self.pod_con_num * self.con_thresh_percent
                ):
                    danger_flag += 1
        R_e = no_effective_flag / self.ser_num  # 低效服务的比例
        R_d = danger_flag / self.ser_num  # 危险服务的比例
        R_b = self.port_num  # 服务中断次数
        R_c = pod_con_num / (pod_num * self.pod_con_num)  # 服务连接数占比
        
        print("ser_num", self.ser_num)

        return self.state, defence_state, defence_success, defence_fail_msg, R_e, R_d, R_b, R_c

    def get_state_index(self, port):
        return self.state[:, 2].tolist().index(port)

    def get_attack_index(self, port):
        return self.attack_state[:, 0].tolist().index(port)
