import numpy as np
from constants import AttackerType, DefenceStrategy


class SCAttacker:
    def __init__(self, env, num):
        self.env = env
        self.num = num
        self.type = AttackerType.SC
        # self.reset()

    def reset(self):
        self.attack_ability = self.num * 256  # 攻击者的能力，能攻陷多少pod
        self.attack_remain = self.num * 256  # 攻击者剩余的资源
        self.target = 0  # 被攻击的目标
        self.target_port = 0  # 被攻击的服务端口

        """侦查阶段:攻击者在第一轮建立观测矩阵"""
        # 要根据port，来确定时延及其他参数
        for port in self.env.state[
            :, 2
        ]:  # 再添加state中新增加的服务:只修改端口号、时延、权重
            ind_s = self.env.get_state_index(port)
            if port > 0:
                for i in range(self.env.ser_max_num):
                    if self.env.attack_state[i][0] == 0:
                        self.env.attack_state[i][0] = self.env.state[ind_s][
                            2
                        ]  # 攻击者探测到的服务端口号
                        # attack_state是int，时延需要扩大100倍才能体现差异
                        self.env.attack_state[i][1] = (
                            100
                            * self.env.state[ind_s][1]
                            / (self.env.state[ind_s][0] * self.env.pod_con_num)
                        )  # 用服务连接数除以服务可承载连接数表示服务时延
                        self.env.attack_state[i][3] = 0.9 * self.env.attack_state[i][
                            1
                        ] + 0.1 * 100 * (
                            self.env.attack_state[i][2]
                            / (self.env.steps_beyond_terminated + 1)
                        )  # 计算服务被攻击的权重
                        break

        # 攻击目标选择
        self.target = np.argmax(
            self.env.attack_state[:, 1]
        )  # 选择时延最高的服务（服务负载率最高）
        self.target_port = self.env.attack_state[self.target][0]  # 被攻击的服务端口号
        target_ser_num = self.env.get_state_index(
            self.target_port
        )  # 在state中找到被攻击的服务序号，因为state和attack_state是通过port连接

        # 开始攻击，根据port分配攻击流量
        if self.attack_remain <= 0:  # 攻击者没有流量就静止
            None
        elif self.attack_remain <= (
            self.env.state[target_ser_num][0] * self.env.pod_con_num
            - self.env.state[target_ser_num][1]
        ):  # 攻击者流量不足以使服务满载，就将剩余流量全部给出
            self.env.state[target_ser_num][1] += self.attack_remain
            self.env.attack_state[self.target][4] += self.attack_remain
            self.attack_remain = 0
            self.env.attack_state[self.target][2] += 1
        else:
            self.env.attack_state[self.target][4] += (
                self.env.state[target_ser_num][0] * self.env.pod_con_num
                - self.env.state[target_ser_num][1]
            )
            self.attack_remain -= (
                self.env.state[target_ser_num][0] * self.env.pod_con_num
                - self.env.state[target_ser_num][1]
            )
            self.env.attack_state[self.target][2] += 1
            self.env.state[target_ser_num][1] = (
                self.env.state[target_ser_num][0] * self.env.pod_con_num
            )  # 使被攻击的服务满载

        return np.array(self.env.state, dtype=np.int64), {}

    def step(self, defence_strategy):
        """攻击者模型"""
        # 防御动作后，攻击者流量的变化
        if (
            defence_strategy == DefenceStrategy.PORT_HOPPING
        ):  # 发生端口变换的服务攻击流量要回收
            for port in self.env.port_list:
                if port in self.env.attack_state[:, 0]:
                    ind = self.env.get_attack_index(port)
                    self.attack_remain += self.env.attack_state[ind][4]
                    self.env.attack_state[ind][4] = 0
        elif (
            defence_strategy == DefenceStrategy.REPLICA_INCREASE
        ):  # 增加副本，攻击流量需要分配给新副本一半，需要在attack_state中添加新的服务
            for port in self.env.add_ser_list1:
                if port in self.env.attack_state[:, 0]:
                    ind = self.env.get_attack_index(port)
                    tmp = 0.5 * self.env.attack_state[ind][4]
                    self.env.attack_state[ind][4] = tmp
                    new_port = self.env.add_ser_list2[
                        self.env.add_ser_list1.index(port)
                    ]
                    for i in range(self.env.ser_max_num):
                        if self.env.attack_state[i][0] == 0:
                            self.env.attack_state[i][0] = new_port
                            self.env.attack_state[i][4] = tmp
                            break
        elif defence_strategy == DefenceStrategy.REPLICA_DECREASE:
            for port in self.env.del_ser_list:
                if port in self.env.attack_state[:, 0]:
                    ind = self.env.get_attack_index(port)
                    attack_con = self.env.attack_state[ind][4]
                    self.env.attack_state[ind][4] = 0
                    for i in range(self.env.ser_max_num):
                        if self.env.attack_state[i][0]:
                            self.env.attack_state[i][4] += (
                                attack_con // self.env.ser_num
                            )
                            break

        """ 侦查阶段:攻击者在第一轮建立观测矩阵,后面只需要添加或者删除port以及对应的服务;防御方执行端口变换，攻击者静默一轮，不攻击 """
        if self.env.port_list == []:
            # 要根据port，来确定时延及其他参数
            for port in self.env.attack_state[
                :, 0
            ]:  # 先删除state里已经不存在的port，全部赋值为0
                if port not in self.env.state[:, 2]:
                    ind = self.env.get_attack_index(port)
                    self.env.attack_state[ind] = np.array([0, 0, 0, 0, 0])
            for port in self.env.state[
                :, 2
            ]:  # 再添加state中新增加的服务:只修改端口号、时延、权重
                ind_s = self.env.get_state_index(port)
                if port > 0:
                    if port in self.env.attack_state[:, 0]:
                        ind = self.env.get_attack_index(port)
                        self.env.attack_state[ind][0] = self.env.state[ind_s][
                            2
                        ]  # 攻击者探测到的服务端口号
                        # 时延太小，取整后差别无法体现，扩大100倍；
                        self.env.attack_state[ind][1] = (
                            100
                            * self.env.state[ind_s][1]
                            / (self.env.state[ind_s][0] * self.env.pod_con_num)
                        )  # 用服务连接数除以服务可承载连接数表示服务时延
                        self.env.attack_state[ind][3] = 0.9 * self.env.attack_state[
                            ind
                        ][1] + 0.1 * 100 * (
                            self.env.attack_state[ind][2]
                            / (self.env.steps_beyond_terminated + 1)
                        )  # 计算服务被攻击的权重
                    else:
                        for i in range(self.env.ser_max_num):
                            if self.env.attack_state[i][0] == 0:
                                self.env.attack_state[i][0] = self.env.state[ind_s][
                                    2
                                ]  # 攻击者探测到的服务端口号
                                # attack_state是int，时延需要扩大100倍才能体现差异
                                self.env.attack_state[i][1] = (
                                    100
                                    * self.env.state[ind_s][1]
                                    / (self.env.state[ind_s][0] * self.env.pod_con_num)
                                )  # 用服务连接数除以服务可承载连接数表示服务时延
                                self.env.attack_state[i][
                                    3
                                ] = 0.9 * self.env.attack_state[i][1] + 0.1 * 100 * (
                                    self.env.attack_state[i][2]
                                    / (self.env.steps_beyond_terminated + 1)
                                )  # 计算服务被攻击的权重
                                break

            # 攻击目标选择
            if self.env.steps_beyond_terminated == 1:
                self.target = np.argmax(
                    self.env.attack_state[:, 1]
                )  # 选择时延最高的服务
                self.target_port = self.env.attack_state[self.target][
                    0
                ]  # 被攻击的服务端口号
            elif self.target_port not in self.env.attack_state[:, 0]:
                self.target = np.argmax(
                    self.env.attack_state[:, 1]
                )  # 选择时延最高的服务
                self.target_port = self.env.attack_state[self.target][
                    0
                ]  # 被攻击的服务端口号
            target_ser_num = self.env.get_state_index(
                self.target_port
            )  # 在state中找到被攻击的服务序号，因为state和attack_state是通过port连接

            # 开始攻击，根据port分配攻击流量
            if self.attack_remain <= 0:
                None
            elif self.attack_remain <= (
                self.env.state[target_ser_num][0] * self.env.pod_con_num
                - self.env.state[target_ser_num][1]
            ):
                self.env.state[target_ser_num][1] += self.attack_remain
                self.env.attack_state[self.target][4] += self.attack_remain
                self.attack_remain = 0
                self.env.attack_state[self.target][2] += 1
            else:
                self.env.attack_state[self.target][4] += (
                    self.env.state[target_ser_num][0] * self.env.pod_con_num
                    - self.env.state[target_ser_num][1]
                )
                self.attack_remain -= (
                    self.env.state[target_ser_num][0] * self.env.pod_con_num
                    - self.env.state[target_ser_num][1]
                )
                self.env.attack_state[self.target][2] += 1
                self.env.state[target_ser_num][1] = (
                    self.env.state[target_ser_num][0] * self.env.pod_con_num
                )  # 使被攻击的服务满载
