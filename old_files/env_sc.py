''' Attackers with a single and constant (S & C) target --  S & C Scenario '''
import gym
from gym import spaces
import numpy as np
from typing import Optional
import math

class EnvSC(gym.Env):
    def __init__(self, att_num=None, render_mode: Optional[str] = None):
        self.pod_max_num = 100 # 边缘节点总的计算资源
        self.pod_con_num = 256 # 单个pod最大连接数
        self.ser_max_num = 10 # 最大副本数量
        self.ser_ind = 3 # 服务副本的子指标数量
        self.ser_num = 0 # 当前服务副本的数量
        if att_num is None:
            self.att_num = 10
        else:
            self.att_num = att_num # 攻击者数量，默认为10，还可以是20，30，40，50
        self.con_thresh_percent = 0.75 # 正常服务连接数量占比阈值
        self.alpha, self.beta, self.gamma, self.delta = 8, 1, 0.02, 0.5 # 奖励计算权重

        high = np.zeros((self.ser_max_num, self.ser_ind), dtype=np.int64)
        low = np.zeros((self.ser_max_num, self.ser_ind), dtype=np.int64)
        for i in range(self.ser_max_num):
            high[i] = [100, 25600, 32767]
            low[i] = [0, 0, 30000]

        self.action_space = spaces.Discrete(6) # 动作空间的大小，一维
        self.observation_space = spaces.Box(low, high, shape=(self.ser_max_num, self.ser_ind), dtype=np.int64) # Box（10，3）

    def get_state_index(self, port ):
        return self.state[:,2].tolist().index(port)

    def get_attack_index(self, port):
        return self.attack_state[:,0].tolist().index(port)

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        pod_remain = self.pod_max_num - np.sum(self.state[:, 0])# 剩余pod的数量即计算资源
        reward = None # 奖励
        break_time, port_num = -0.1, 0
        self.for_state = self.state.copy() # 保存前一轮state，采用copy()方法深拷贝
        noaction_pen = -1 # 执行动作01234，但是没有采取实质行动的惩罚
        port_pen = 0 # 端口变换发生在资源充足时的惩罚
        port_list = [] # 记录攻击者攻击后，进行端口变换的服务的原来的port
        add_ser_list1 = [] # 扩展副本的服务
        add_ser_list2 = [] # 扩展副本产生的新服务
        del_ser_list = [] # 被删除副本的服务
        
        '''
        action执行动作->环境发生状态转移->得到reward
        '''
        if action == 0: # 端口变换
            ''' 当防御资源不充足的条件下，使用端口变换，达到短暂的防御效果 '''
            if pod_remain <= 5:
                for i in range(self.ser_max_num):
                    #if self.state[i][1] > self.con_thresh_percent * self.state[i][0] * self.pod_con_num:
                    if self.state[i][0] > 0:
                        port_list.append(self.state[i][2])
                        if self.state[i][2] in self.attack_state[:,0]:
                            ind = self.get_attack_index(self.state[i][2])
                            self.state[i][1] = self.state[i][1] - self.attack_state[ind][4]                    
                        port = np.random.randint(30000, 32767)
                        while port in self.state[:, 2]: # 确保端口号不与原来的端口和其他已使用端口重复
                            port = np.random.randint(30000, 32767)
                        self.state[i][2] = port
                        port_num += 1
                        noaction_pen = 0
            else:
                port_pen = -1
        elif action == 1: # 副本增加
            '''
            选定服务过载的副本,副本的pod数量和原服务一样,并将所有流量的1/2给副本;
            服务数量和pod数量达到最大时，不能选择1和3，否则施加惩罚;
            '''
            if self.ser_num == self.ser_max_num or pod_remain == 0:
                noaction_pen = -1
            else:
                for i in range(self.ser_max_num):
                    if self.state[i][1] > self.con_thresh_percent * self.state[i][0] * self.pod_con_num:
                        if pod_remain >= self.state[i][0] and self.ser_num < self.ser_max_num:
                            add_ser_list1.append(self.state[i][2])
                            new_pod = self.state[i][0]
                            connection = 0.5 * self.state[i][1]
                            self.state[i][1] = connection
                            port = np.random.randint(30000, 32767)
                            while port in self.state[:, 2]:  
                                port = np.random.randint(30000, 32767)                         
                            for j in range(self.ser_max_num): # 给扩展的副本定位位置
                                if self.state[j][0] == 0:
                                    self.state[j] = np.array([new_pod, connection, port])
                                    add_ser_list2.append(port) 
                                    pod_remain -= new_pod                                    
                                    self.ser_num += 1
                                    noaction_pen = 0
                                    break                      
                        else:
                            noaction_pen = -1 
        elif action == 2: # 减少副本
            '''
            对单个pod的服务进行副本的删除,删除的服务的流量(包括用户和攻击者)平摊到各个其他服务；
            服务数量不能小于1；
            '''
            if self.ser_num <= 1:
                noaction_pen = -1
            else:
                for i in range(self.ser_max_num):
                    if self.state[i][0] == 1:
                        con_num = self.state[i][1]                        
                        self.state[i] = np.array([0, 0, 0])  
                        self.ser_num -= 1  
                        pod_remain += 1
                        noaction_pen = 0
                        for j in range(self.ser_max_num):
                            if self.state[j][0] and j != i:
                                self.state[j][1] += con_num // self.ser_num           
        elif action == 3: # 副本扩容
            '''
            选择负载率大于0.75的服务全部进行扩容,扩容后负载率为75%, 保证服务质量;
            '''
            if pod_remain == 0:
                noaction_pen = -1
                #res_flag = True 
            else:
                for i in range(self.ser_max_num):
                    if self.state[i][1] > self.con_thresh_percent * self.state[i][0] * self.pod_con_num:
                        pod_incre = int(math.ceil(self.state[i][1] / (self.pod_con_num*self.con_thresh_percent) - self.state[i][0]))
                        if  pod_remain >= pod_incre:
                            self.state[i][0] = self.state[i][0] + pod_incre
                            pod_remain -= pod_incre
                            noaction_pen = 0 
                        else:
                            None
        elif action == 4: # 副本缩容
            '''
            选择负载率小于0.5的服务全部进行缩容,缩容后负载率为75%，保证能耗比最低；
            '''    
            for i in range(self.ser_max_num):
                if self.state[i][1] < 0.5 * self.state[i][0] * self.pod_con_num:
                    pod_decre = int(self.state[i][0] - self.state[i][1] / (self.pod_con_num*self.con_thresh_percent))
                    self.state[i][0] = self.state[i][0] - pod_decre
                    pod_remain += pod_decre
                    noaction_pen = 0         
        elif action == 5: # 静止动作
            noaction_pen = 0
            for i in range(self.ser_max_num):
                if self.state[i][1] > self.con_thresh_percent * self.state[i][0] * self.pod_con_num: 
                    noaction_pen = -1
                    break

        # print("after defense:", self.state)

        ''' 攻击者模型 '''
        # 防御动作后，攻击者流量的变化
        if action == 0: # 发生端口变换的服务攻击流量要回收
            for port in port_list: 
                if port in self.attack_state[:, 0]:
                    ind = self.get_attack_index(port)
                    self.attack_remain += self.attack_state[ind][4]
                    self.attack_state[ind][4] = 0 
        elif action == 1: # 增加副本，攻击流量需要分配给新副本一半，需要在attack_state中添加新的服务
            for port in add_ser_list1:
                if port in self.attack_state[:, 0]:
                    ind = self.get_attack_index(port)
                    tmp = 0.5 * self.attack_state[ind][4]
                    self.attack_state[ind][4] = tmp
                    new_port = add_ser_list2[add_ser_list1.index(port)]
                    for i in range(self.ser_max_num):
                        if self.attack_state[i][0] == 0:
                            self.attack_state[i][0] = new_port
                            self.attack_state[i][4] = tmp
                            break  
        elif action == 2:
            for port in del_ser_list:
                if port in self.attack_state[:, 0]:
                    ind = self.get_attack_index(port)
                    attack_con = self.attack_state[ind][4]
                    self.attack_state[ind][4] = 0
                    for i in range(self.ser_max_num):
                        if self.attack_state[i][0]:
                            self.attack_state[i][4] += attack_con // self.ser_num
                            break                    

        ''' 侦查阶段:攻击者在第一轮建立观测矩阵,后面只需要添加或者删除port以及对应的服务;防御方执行端口变换，攻击者静默一轮，不攻击 '''
        #if action != 0:
        if port_list == []:
            # 要根据port，来确定时延及其他参数
            for port in self.attack_state[:, 0]: # 先删除state里已经不存在的port，全部赋值为0
                if port not in self.state[:, 2]:
                        ind = self.get_attack_index(port)
                        self.attack_state[ind] =  np.array([0, 0, 0, 0, 0])        
            for port in self.state[:, 2]: # 再添加state中新增加的服务:只修改端口号、时延、权重
                ind_s = self.get_state_index(port)
                if port > 0:
                    if port in self.attack_state[:, 0]:
                        ind = self.get_attack_index(port)
                        self.attack_state[ind][0] = self.state[ind_s][2] # 攻击者探测到的服务端口号
                        # 时延太小，取整后差别无法体现，扩大100倍；
                        self.attack_state[ind][1] = 100 * self.state[ind_s][1] / (self.state[ind_s][0] * self.pod_con_num) # 用服务连接数除以服务可承载连接数表示服务时延
                        self.attack_state[ind][3] = 0.9 * self.attack_state[ind][1] + 0.1 * 100 * (self.attack_state[ind][2] / (self.steps_beyond_terminated + 1)) # 计算服务被攻击的权重
                    else:
                        for i in range(self.ser_max_num):
                            if self.attack_state[i][0] == 0:
                                self.attack_state[i][0] = self.state[ind_s][2] # 攻击者探测到的服务端口号
                                # attack_state是int，时延需要扩大100倍才能体现差异
                                self.attack_state[i][1] = 100 * self.state[ind_s][1] / (self.state[ind_s][0] * self.pod_con_num) # 用服务连接数除以服务可承载连接数表示服务时延
                                self.attack_state[i][3] = 0.9 * self.attack_state[i][1] + 0.1 * 100 * (self.attack_state[i][2] / (self.steps_beyond_terminated + 1)) # 计算服务被攻击的权重
                                break                    

            # 攻击目标选择
            if self.steps_beyond_terminated == 1:
                self.target = np.argmax(self.attack_state[:, 1]) # 选择时延最高的服务       
                self.target_port = self.attack_state[self.target][0] # 被攻击的服务端口号
            elif self.target_port not in self.attack_state[:, 0]:
                self.target = np.argmax(self.attack_state[:, 1]) # 选择时延最高的服务       
                self.target_port = self.attack_state[self.target][0] # 被攻击的服务端口号
            target_ser_num = self.get_state_index(self.target_port) # 在state中找到被攻击的服务序号，因为state和attack_state是通过port连接

            # 开始攻击，根据port分配攻击流量
            if self.attack_remain <= 0:
                None 
            elif self.attack_remain <= (self.state[target_ser_num][0] * self.pod_con_num - \
                                        self.state[target_ser_num][1]):
                self.state[target_ser_num][1] += self.attack_remain
                self.attack_state[self.target][4] += self.attack_remain
                self.attack_remain = 0
                self.attack_state[self.target][2] += 1
            else:
                self.attack_state[self.target][4] += self.state[target_ser_num][0] * self.pod_con_num - self.state[target_ser_num][1] 
                self.attack_remain -= self.state[target_ser_num][0] * self.pod_con_num - self.state[target_ser_num][1] 
                self.attack_state[self.target][2] += 1            
                self.state[target_ser_num][1] = self.state[target_ser_num][0] * self.pod_con_num # 使被攻击的服务满载
        
        # reward奖励函数
        # 第四版：将这一时刻状态与前一时刻对比，得到收益
        success_flag = 0
        for i in range(self.ser_max_num):
            if self.state[i][0] > 0 and self.state[i][1] <= self.con_thresh_percent * self.state[i][0] * self.pod_con_num: 
                success_flag += 1
        sum, num = 0, 0
        for i in range(self.ser_max_num):
            if self.for_state[i][0] and self.state[i][0]:
                sum += (self.for_state[i][1] / (self.for_state[i][0] * self.pod_con_num) - \
                        self.state[i][1] / (self.state[i][0] * self.pod_con_num))
                num += 1
        R_c = sum / num
        R_s = (np.sum(self.for_state[:, 0]) - np.sum(self.state[:, 0]) ) / self.pod_max_num
        R_t = break_time * port_num
        for_ser_num = 0
        for i in range(self.ser_max_num):
            if self.for_state[i][0]:
                for_ser_num += 1
        R_a = (self.ser_num - for_ser_num) / self.ser_num
        reward = self.alpha * R_c +  self.beta * R_s + self.gamma * R_a + self.delta * R_t + noaction_pen + port_pen + \
                success_flag / self.ser_num    
 
        # episode中止条件
        # 条件一：每个服务的已有连接数不能大于本身服务能承载的连接数
        con_flag = False
        for i in range(self.ser_max_num):
            if not self.state[i][0]:
                self.state[i][1] > self.state[i][0] * self.pod_con_num
                con_flag = False
        # 条件二：剩余的pod数量要不小于0
        pod_flag = bool(pod_remain < 0)
        # 条件三：服务的连接数不小于0
        ser_con_flag = bool(np.min(self.state[:, 1]) < 0)
        terminated = bool(
            pod_flag
            or ser_con_flag
            or con_flag
        )

        if terminated and self.steps_beyond_terminated < 20:
            reward -= 1
        
        self.steps_beyond_terminated += 1 # 限制agent和环境交互的次数，因为攻防博弈没有确定停止的点


        return np.array(self.state, dtype=np.int64), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):  

        self.state = None # 状态矩阵
        self.for_state = None # 前一时刻的状态矩阵
        self.attack_state = np.zeros((self.ser_max_num, 5), dtype=np.int64) # 攻击者可以观测环境中的信息矩阵:服务端口号，负载率（时延），被攻击次数，攻击权重，攻击流量
        self.attack_ability = self.att_num * 256 # 攻击者的能力，能攻陷多少pod
        self.attack_remain = self.att_num * 256 # 攻击者剩余的资源
        self.target = 0 # 被攻击的目标
        self.target_port = 0 # 被攻击的服务端口

        '''
        仿真环境:初始为pod总量一半的50个pod,用户总连接数定义在pod总数*256的1/2到3/4;
                优质服务质量为小于3/4的连接总数;
        '''
        self.state = np.zeros((self.ser_max_num, self.ser_ind ), dtype=np.int64)
        self.ser_num = 5
        for i in range(self.ser_num):
            port = np.random.randint(30000, 32767)
            while port in self.state[:, 2]: 
                port = np.random.randint(30000, 32767)
            connection = np.random.randint(int(10*256*0.5), int(10*256*self.con_thresh_percent))
            self.state[i] = [10, connection, port]
        self.steps_beyond_terminated = 0

        ''' 侦查阶段:攻击者在第一轮建立观测矩阵 '''
        # 要根据port，来确定时延及其他参数     
        for port in self.state[:, 2]: # 再添加state中新增加的服务:只修改端口号、时延、权重
            ind_s = self.get_state_index(port)
            if port > 0:
                for i in range(self.ser_max_num):
                    if self.attack_state[i][0] == 0:
                        self.attack_state[i][0] = self.state[ind_s][2] # 攻击者探测到的服务端口号
                        # attack_state是int，时延需要扩大100倍才能体现差异
                        self.attack_state[i][1] = 100 * self.state[ind_s][1] / (self.state[ind_s][0] * self.pod_con_num) # 用服务连接数除以服务可承载连接数表示服务时延
                        self.attack_state[i][3] = 0.9 * self.attack_state[i][1] + 0.1 * 100 * (self.attack_state[i][2] / (self.steps_beyond_terminated + 1)) # 计算服务被攻击的权重
                        break                    

        # 攻击目标选择
        self.target = np.argmax(self.attack_state[:, 1]) # 选择时延最高的服务（服务负载率最高）       
        self.target_port = self.attack_state[self.target][0] # 被攻击的服务端口号
        target_ser_num = self.get_state_index(self.target_port) # 在state中找到被攻击的服务序号，因为state和attack_state是通过port连接

        # 开始攻击，根据port分配攻击流量
        if self.attack_remain <= 0: # 攻击者没有流量就静止
            None 
        elif self.attack_remain <= (self.state[target_ser_num][0] * self.pod_con_num - \
                                    self.state[target_ser_num][1]): # 攻击者流量不足以使服务满载，就将剩余流量全部给出
            self.state[target_ser_num][1] += self.attack_remain
            self.attack_state[self.target][4] += self.attack_remain
            self.attack_remain = 0
            self.attack_state[self.target][2] += 1
        else:
            self.attack_state[self.target][4] += self.state[target_ser_num][0] * self.pod_con_num - self.state[target_ser_num][1] 
            self.attack_remain -= self.state[target_ser_num][0] * self.pod_con_num - self.state[target_ser_num][1] 
            self.attack_state[self.target][2] += 1            
            self.state[target_ser_num][1] = self.state[target_ser_num][0] * self.pod_con_num # 使被攻击的服务满载

        return np.array(self.state, dtype=np.int64), {}

if __name__ == "__main__":
    a = EnvSC()

    state, _ = a.reset()
    print(a.state)
    print(a.attack_state)