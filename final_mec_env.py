#  mec环境简单版
#  M个物联网设备和N个MEC服务器，一个中继器， 设备将任务发送给中继器，中继器针对任务的性质
#  状态state
#           本地cpu频率 F_Local = {f1, f2, ... fm}   离散
#           服务器选择 C_MEC = {cm1, cm2 ... cmm} 离散
#           发送功率 P_send = {p1, p2, ... pm} ,    连续
#           每个任务的卸载率X = {x1,x2,...xm}，  连续
#
#           卸载率和发送功率是连续的，本地CPU频率和服务器选择是离散的
#
#           State = {X, F_Local, P_send,  C_MEC}
#  动作action Action = {ΔX, ΔF_Local, ΔP_send, ΔC, ΔQ}
import math

import numpy as np

num_d = 50  # 任务数
num_m = 4  # 服务器数量
num_q_tasks = 4  # 任务性质
num_c = 13  # 信道选择
h = 0.98
v = 4.
N = 10 ** -11
alpha_k = np.random.uniform(10, 20, num_d)
w = 10 ** 8
yita = 1.2
lamb = 10 ** (-26)
t_threshold = 1000
k_t = 0
k_e = 1
f_index = 10 ** 8
mec_max_cpu = 12  # mec内核数
mec_q_acc = 1.2  # 同类型任务计算加速参数

offload_x_up = 1
offload_x_down = 0
local_f_up = 8  # 本地cpu频率的边界，别忘了乘10^9
local_f_down = 2
p_send_up = 0.1
p_send_down = 0.05
channel_c_up = 13
channel_c_down = 0.00001  # 取0是为了方便向上取整
mec_c_up = 4
mec_c_down = 0.00001  # 取0是为了方便向上取整

np.seterr(divide='ignore', invalid='ignore')


class ENV(object):
    def __init__(self, num_device, offload_x, local_f, mec_f, send_p, channel_c, d, b, num_mec,
                 task_quality, mec_c, t_threshold, mec_max_cpu, mec_q_acc, task_priority):
        self.num_device = num_device  # num_device
        self.offload_x = offload_x  # num_device
        self.local_f = local_f  # num_device
        self.mec_f = mec_f  # num_mec
        self.send_p = send_p  # num_device
        self.channel_c = channel_c  # m
        self.d = d  # 1 * k 到mec的距离  [num_device, num_mec]
        self.b = b  # 1 * k 任务量  num_device
        self.t_threshold = t_threshold  # num_device
        self.num_mec = num_mec  # num_mec
        self.task_quality = task_quality  # num_q_tasks
        self.mec_c = mec_c  # num_mec
        self.mec_max_cpu = mec_max_cpu  # 12
        self.mec_q_acc = mec_q_acc  # 1.2
        self.task_priority = task_priority  # num_device
        self.t_all = []
        self.e_all = []
        self.cost_all = []
        self.u = []

    def calculate_co_channel_interference(self):
        np.array(self.offload_x)
        # nkc = np.zeros(self.num_device)
        # for i in range(self.num_device):
        #     for c in range(self.num_device):
        #         if (self.channel_c[i] == self.channel_c[c]) & (i != c):
        #             nkc[i] += (self.d[c][self.mec_c[c] - 1] ** (-v)) * (h ** 2) * self.send_p[i]
        d_mec = np.zeros(self.num_device)
        for i in range(self.num_device):
            d_mec[i] = self.d[i][self.mec_c[i] - 1]
        co_channel_interference = np.divide(self.send_p * (d_mec ** (-v)) * (h ** 2), N)  # 应该大于1
        return co_channel_interference

    def calculate_all_delay(self):
        t_local = np.divide(alpha_k * (1 - self.offload_x) * self.b, self.local_f)
        sinr = self.calculate_co_channel_interference()
        r = w * np.log2(1 + sinr)
        t_communication = np.divide(yita * self.offload_x * self.b, r)
        # print(t_communication)
        # 进入同一个mec服务器时 最多能同时处理的任务数量是 12个，所以会出现任务排队的情况，假设任务处理按照优先级进行处理
        mec1 = []
        mec2 = []
        mec3 = []
        mec4 = []
        for i in range(self.num_device):
            if self.mec_c[i] == 0:
                mec1.append(i)
            if self.mec_c[i] == 1:
                mec2.append(i)
            if self.mec_c[i] == 2:
                mec3.append(i)
            if self.mec_c[i] == 3:
                mec4.append(i)
        t_mec = np.zeros(self.num_device)
        t_mec1 = self.calculate_sub_mec_delay(mec1, 0)
        t_mec2 = self.calculate_sub_mec_delay(mec2, 1)
        t_mec3 = self.calculate_sub_mec_delay(mec3, 2)
        t_mec4 = self.calculate_sub_mec_delay(mec4, 3)
        for i in range(self.num_device):
            if t_mec1[i] != 0:
                t_mec[i] = t_mec1[i]
            if t_mec2[i] != 0:
                if t_mec[i] != 0:
                    print("error")
                t_mec[i] = t_mec2[i]
            if t_mec3[i] != 0:
                if t_mec[i] != 0:
                    print("error")
                t_mec[i] = t_mec3[i]
            if t_mec4[i] != 0:
                if t_mec[i] != 0:
                    print("error")
                t_mec[i] = t_mec4[i]
        for i in range(self.num_device):
            if self.task_quality[i] == self.mec_c[i]:
                t_mec[i] /= 1.2
        t_all = np.zeros(num_d)
        for i in range(num_d):
            t_all[i] = max(t_local[i], t_communication[i] + t_mec[i])
        return t_all, t_communication

    def calculate_sub_mec_delay(self, mec_i, mec_n):
        t_mec = np.zeros(self.num_device)
        if len(mec_i) <= self.mec_max_cpu:
            for mi in mec_i:
                t_mec[mi] = np.divide(alpha_k[mi] * self.offload_x[mi] * self.b[mi], self.mec_f[mec_n])
        else:
            for i in mec_i:
                for j in mec_i:
                    if self.task_priority[i] < self.task_priority[j]:
                        temp = j
                        j = i
                        i = temp
            for i in range(len(mec_i)):
                if i < self.mec_max_cpu:
                    t_mec[mec_i[i]] = np.divide(alpha_k[mec_i[i]] * self.offload_x[mec_i[i]] * self.b[mec_i[i]],
                                                self.mec_f[mec_n])
                else:
                    t_mec[mec_i[i]] = t_mec[mec_i[2 * self.mec_max_cpu - 1 - i]] + \
                                      np.divide(alpha_k[mec_i[i]] * self.offload_x[mec_i[i]] * self.b[mec_i[i]],
                                                self.mec_f[mec_n])
        return t_mec

    def calculate_all_energy(self, t_communication):
        # 先计算总时延，里面有个信道传输时间需要用到
        e_local = lamb * (self.local_f ** 2) * alpha_k * (1 - self.offload_x) * self.b
        e_send = self.send_p * 18 * t_communication
        e_all = e_send + e_local
        return e_all

    def calculate_comprehensive_cost(self):
        t_all, t_communication = self.calculate_all_delay()
        e_all = self.calculate_all_energy(t_communication)
        cost_all = k_t * t_all + k_e * e_all
        return t_all, e_all, cost_all

    #  反归一化，将0-1之间的数转化成符合环境的参数
    def calculate_normalized_parameter(self, offload_x_, local_f_, p_send_, mec_c_):  # mec中的f在（0，1）之间，归一化处理
        self.offload_x = offload_x_
        self.local_f = local_f_ * (local_f_up - local_f_down) + local_f_down
        self.local_f = self.local_f * f_index
        self.send_p = p_send_ * (p_send_up - p_send_down) + p_send_down
        # self.channel_c = np.ceil(channel_c_ * (channel_c_up - channel_c_down) + channel_c_down)
        self.mec_c = np.ceil(mec_c_ * (mec_c_up - mec_c_down) + mec_c_down)
        # self.channel_c = list(map(int, self.channel_c))
        self.mec_c = list(map(int, self.mec_c))

    def calculate_cost_by_state(self, s):
        offload_x_ = s[0: self.num_device]  # 卸载率
        local_f_ = s[self.num_device: 2 * self.num_device]  # 本地cpu频率
        p_send_ = s[self.num_device * 2: 3 * self.num_device]  # 发送功率
        mec_c_ = s[self.num_device * 3: 4 * self.num_device]  # 发送至服务器n
        self.calculate_normalized_parameter(offload_x_, local_f_, p_send_, mec_c_)

        t_all, e_all, cost_all = self.calculate_comprehensive_cost()

        e_all = np.sum(e_all)
        t_all = np.sum(t_all)
        cost_all = np.sum(cost_all)
        total_ab = np.sum(self.b) / 10e7
        u = total_ab - 0.5 * e_all - 0.5 * t_all

        return e_all, t_all, u

    def step(self, s, action):  # action [-1,1] && a + s [0,1]
        action = np.array(action)
        s_ = s + action
        offload_x_ = s_[0: self.num_device]  # 卸载率
        local_f_ = s_[self.num_device: 2 * self.num_device]  # 本地cpu频率
        p_send_ = s_[self.num_device * 2: 3 * self.num_device]  # 发送功率
        # channel_c_ = s_[self.num_device * 3: 4 * self.num_device]  # 信道选择
        mec_c_ = s_[self.num_device * 3: 4 * self.num_device]  # 发送至服务器n

        t_all, e_all, cost_all = self.calculate_comprehensive_cost()

        total_ab = np.sum(self.b) / 10e7
        u = total_ab - 0.5 * np.sum(e_all) - 0.5 * np.sum(t_all)
        #  反归一化
        self.calculate_normalized_parameter(offload_x_, local_f_, p_send_, mec_c_)

        for i in s_:
            if i < 0:
                print("error_0")
            if i > 1:
                print("error_1")

        t_all_, e_all_, cost_all_ = self.calculate_comprehensive_cost()
        self.t_all.append(np.sum(t_all_))
        self.e_all.append(np.sum(e_all_))
        self.cost_all.append(np.sum(cost_all_))

        # r = np.zeros(num_d)
        r = 0
        e_all = np.sum(e_all)
        t_all = np.sum(t_all)
        cost_all = np.sum(cost_all)

        e_all_ = np.sum(e_all_)
        t_all_ = np.sum(t_all_)
        cost_all_ = np.sum(cost_all_)
        total_ab = np.sum(self.b) / 10e7
        u_ = total_ab - 0.5 * e_all_ - 0.5 * t_all_
        self.u.append(u_)
        # print(u)

        # if cost_all <= cost_all_:
        #     r += -10
        # if cost_all > cost_all_:
        #     r += 10

        # r += (cost_all - cost_all_) / num_d

        # r += (np.min(self.cost_all) - cost_all_) * 0.01 / num_d
        r += (u_ - 0.8 * np.max(self.u)) * 0.01 / num_d
        # if u_ <= u:
        #     r += -1
        # if u_ > u:
        #     r += 1

        return s, action, r, s_, e_all_, t_all_, u_

    def step_dqn(self, action):
        # 1 - num_d * 8

        t_all, e_all, cost_all = self.calculate_comprehensive_cost()

        total_ab = np.sum(self.b) / 10e7
        u = total_ab - 0.5 * np.sum(e_all) - 0.5 * np.sum(t_all)

        device_no = math.floor(action / 8)
        delta = action % 8
        if delta == 0:
            self.offload_x[device_no] = np.clip(self.offload_x[device_no] + 0.05, 0, 1)
        if delta == 1:
            self.offload_x[device_no] = np.clip(self.offload_x[device_no] - 0.05, 0, 1)
        if delta == 2:
            self.local_f[device_no] = np.clip(self.local_f[device_no] + 1000000, local_f_down * f_index, local_f_up * f_index)
        if delta == 3:
            self.local_f[device_no] = np.clip(self.local_f[device_no] - 1000000, local_f_down * f_index, local_f_up * f_index)
        if delta == 4:
            self.send_p[device_no] = np.clip(self.send_p[device_no] + 0.0005, p_send_down, p_send_up)
        if delta == 5:
            self.send_p[device_no] = np.clip(self.send_p[device_no] - 0.0005, p_send_down, p_send_up)
        self.mec_c = list(map(int, self.mec_c))
        if delta == 6:
            self.mec_c[device_no] = np.clip(self.mec_c[device_no] + 1, mec_c_down, mec_c_up)
        if delta == 6:
            self.mec_c[device_no] = np.clip(self.mec_c[device_no] - 1, mec_c_down, mec_c_up)
        self.mec_c = list(map(int, self.mec_c))

        t_all_, e_all_, cost_all_ = self.calculate_comprehensive_cost()

        total_ab_ = np.sum(self.b) / 10e7
        u_ = total_ab_ - 0.5 * np.sum(e_all_) - 0.5 * np.sum(t_all_)

        x_ = self.offload_x
        f_ = (self.local_f / f_index - local_f_down) / (local_f_up - local_f_down)
        p_send_ = (self.send_p - p_send_down) / (p_send_up - p_send_down)
        mec_c_ = np.array(self.mec_c) / mec_c_up
        s_ = np.append(x_, p_send_)
        s_ = np.append(s_, f_)
        s_ = np.append(s_, mec_c_)

        self.u.append(u_)
        r = 0
        # r += (u_ - 0.8 * np.max(self.u)) * 0.01 / num_d
        r += (u_ - u) / num_d
        return action, r, s_, e_all_, t_all_, u_


num_device = 50
num_mec = 4
offload_x = np.ones(num_device)  # 归一化
local_f = np.random.uniform(0, 1, num_device) # cpu归一化处理 [0,1] ->[2 * 10^9 , 4 * 10^9]
# local_f = np.ceil(local_f * (10 ** 9))
mec_f = np.array([10 ** 10, 10 ** 10, 10 ** 10, 10 ** 10])
send_p = np.random.uniform(0, 1, num_device)  # 归一化
channel_c = np.random.uniform(0, 1, num_device)  # 归一化
d = np.random.randint(0., 500., (num_device, num_mec))
b = np.random.uniform(100, 1000, num_device)
b = np.ceil(b * 1000 * 1024)
mec_c = np.random.uniform(0, 1, num_device)  # 归一化
task_priority = np.random.uniform(0, 1, num_device)
task_quality = np.random.uniform(0, 1, num_device)  # 归一化
mec = ENV(num_device, offload_x, local_f, mec_f, send_p, channel_c, d, b, num_mec,
          task_quality, mec_c, t_threshold, mec_max_cpu, mec_q_acc, task_priority)

mec.calculate_normalized_parameter(offload_x, local_f, send_p, mec_c)
t_all, e_all, cost_all = mec.calculate_comprehensive_cost()

total_ab = np.sum(mec.b) / 10e7
u = total_ab - 0.5 * np.sum(e_all) - 0.5 * np.sum(t_all)
print(np.average(t_all))
print(np.average(e_all))
print(u)
print(total_ab)