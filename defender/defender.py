import numpy as np
import math
from constants import DefenderType, DefenceStrategy

class Defender:
    def __init__(self, env, defender_type: DefenderType):
        self.type = defender_type
        self.env = env
        
    def reset(self):
        '''
        仿真环境:初始为pod总量一半的50个pod,用户总连接数定义在pod总数*256的1/2到3/4;
                优质服务质量为小于3/4的连接总数;
        '''
        for i in range(self.env.ser_num):
            port = np.random.randint(30000, 32767)
            while port in self.env.state[:, 2]: 
                port = np.random.randint(30000, 32767)
            connection = np.random.randint(int(10*256*0.5), int(10*256*self.env.con_thresh_percent))
            self.env.state[i] = [10, connection, port]
    
    def step(self, defence_strategy, params):
        inf_services = []
        '''
        action执行动作->环境发生状态转移->得到reward
        '''
        if defence_strategy == DefenceStrategy.PORT_HOPPING: # 端口变换
            ''' 当防御资源不充足的条件下，使用端口变换，达到短暂的防御效果 '''
            for i in range(self.env.ser_max_num):
                if self.env.state[i][0] > 0:
                    inf_services.append(i)
                    self.env.port_list.append(self.env.state[i][2])
                    # 重置攻击流量，正常流量不变？
                    if self.env.state[i][2] in self.env.attack_state[:,0]:
                        ind = self.env.get_attack_index(self.env.state[i][2])
                        self.env.state[i][1] = self.env.state[i][1] - self.env.attack_state[ind][4]                    
                    port = np.random.randint(30000, 32767)
                    while port in self.env.state[:, 2]: # 确保端口号不与原来的端口和其他已使用端口重复
                        port = np.random.randint(30000, 32767)
                    self.env.state[i][2] = port
            self.env.port_num += 1
            return True, f"服务副本{inf_services}端口变换成功"
        elif defence_strategy == DefenceStrategy.REPLICA_INCREASE: # 副本增加
            '''
            选定超过指定负载率的服务副本增加副本,副本的pod数量和原服务一样,并将所有流量的1/2给副本;
            服务数量和pod数量达到最大时，不能选择1和3，否则施加惩罚;
            '''
            if self.env.ser_num == self.env.ser_max_num:
                return False, "副本数量已经达到最大"
            elif self.env.pod_remain == 0:
                return False, "没有剩余的资源节点供分配"
            else:
                con_percent = params["con_percent"]
                for i in range(self.env.ser_max_num):
                    if self.env.state[i][1] > con_percent * self.env.state[i][0] * self.env.pod_con_num:
                        if self.env.pod_remain >= self.env.state[i][0] and self.env.ser_num < self.env.ser_max_num:
                            inf_services.append(i)
                            self.env.add_ser_list1.append(self.env.state[i][2])
                            new_pod = self.env.state[i][0]
                            connection = 0.5 * self.env.state[i][1]
                            self.env.state[i][1] = connection
                            port = np.random.randint(30000, 32767)
                            while port in self.env.state[:, 2]:  
                                port = np.random.randint(30000, 32767)                         
                            for j in range(self.env.ser_max_num): # 给扩展的副本定位位置
                                if self.env.state[j][0] == 0:
                                    self.env.state[j] = np.array([new_pod, connection, port])
                                    self.env.add_ser_list2.append(port) 
                                    self.env.pod_remain -= new_pod                                    
                                    self.env.ser_num += 1
                                    break
                if len(inf_services) == 0:
                    return False, "没有剩余的资源节点供分配"
                else:
                    return True, f"服务{inf_services}副本增加成功"
        elif defence_strategy == DefenceStrategy.REPLICA_DECREASE: # 减少副本
            '''
            对小于指定负载率的副本进行删除,删除的服务的流量(包括用户和攻击者)平摊到各个其他服务；
            服务数量不能小于1；
            '''
            con_percent = params["con_percent"]
            if self.env.ser_num <= 1 or self.env.state[:][1].max() < con_percent * self.env.state[:][0].max() * self.env.pod_con_num:
                return False, "服务副本数量不能小于1"
            else:
                con_num = 0
                for i in range(self.env.ser_max_num):
                    if self.env.state[i][1] < con_percent * self.env.state[i][0] * self.env.pod_con_num:
                        inf_services.append(i)
                        con_num += self.env.state[i][1]                        
                        self.env.state[i] = np.array([0, 0, 0])  
                        self.env.ser_num -= 1
                        self.env.pod_remain += 1
                for j in range(self.env.ser_max_num):
                    if self.env.state[j][0]:
                        self.env.state[j][1] += con_num // self.env.ser_num
                return True, f"服务{inf_services}副本减少成功"        
        elif defence_strategy == DefenceStrategy.REPLICA_EXPAND: # 副本扩容
            '''
            选择大于指定负载率的服务全部进行扩容,扩容后负载率为指定负载率, 保证服务质量;
            '''
            if self.env.pod_remain == 0:
                return False, "没有剩余的资源节点供扩展"
            else:
                con_percent = params["con_percent"]
                for i in range(self.env.ser_max_num):
                    if self.env.state[i][1] > con_percent * self.env.state[i][0] * self.env.pod_con_num:
                        pod_incre = int(math.ceil(self.env.state[i][1] / (self.env.pod_con_num*con_percent) - self.env.state[i][0]))
                        if  self.env.pod_remain >= pod_incre:
                            inf_services.append(i)
                            self.env.state[i][0] = self.env.state[i][0] + pod_incre
                            self.env.pod_remain -= pod_incre
                if len(inf_services) == 0:
                    return False, "没有剩余的资源节点供扩展"
                else:
                    return True, f"服务{inf_services}副本扩容成功"
        elif defence_strategy == DefenceStrategy.REPLICA_SHRINK: # 副本缩容
            '''
            选择小于指定负载率的服务全部进行缩容,缩容后负载率为指定负载率，保证能耗比最低；
            '''    
            for i in range(self.env.ser_max_num):
                con_percent = params["con_percent"]
                if self.env.state[i][1] < con_percent * self.env.state[i][0] * self.env.pod_con_num:
                    inf_services.append(i)
                    pod_decre = int(self.env.state[i][0] - self.env.state[i][1] / (self.env.pod_con_num*con_percent))
                    self.env.state[i][0] = self.env.state[i][0] - pod_decre
                    self.env.pod_remain += pod_decre
            return True, f"服务{inf_services}副本缩容成功"
        elif defence_strategy == DefenceStrategy.NO_ACTION: # 静止动作
            return True, "不采取行动"
                
        return True, None
