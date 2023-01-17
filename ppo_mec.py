import gym
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import final_mec_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorNet(nn.Module):
    def __init__(self, n_states, n_action, bound):
        super(ActorNet, self).__init__()
        self.n_states = n_states
        self.bound = bound

        self.layer = nn.Sequential(
            nn.Linear(self.n_states, 128),
            nn.ReLU()
        )

        self.mu_out = nn.Linear(128, n_action)
        self.sigma_out = nn.Linear(128, n_action)

    def forward(self, x):
        x = F.relu(self.layer(x))
        mu = self.bound * torch.tanh(self.mu_out(x))
        sigma = F.softplus(self.sigma_out(x))
        return mu, sigma


class CriticNet(nn.Module):
    def __init__(self, n_states, n_actions):
        super(CriticNet, self).__init__()
        self.n_states = n_states

        self.layer = nn.Sequential(
            nn.Linear(self.n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        v = self.layer(x)
        return v


class PPO(nn.Module):
    def __init__(self, n_states, n_actions, bound, args):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.bound = bound
        self.lr = args.lr
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.a_update_steps = args.a_update_steps
        self.c_update_steps = args.c_update_steps
        self.actor_loss = []
        self.critic_loss = []

        self._build()

    def _build(self):
        self.actor_model = ActorNet(n_states, n_actions, bound)
        self.actor_old_model = ActorNet(n_states, n_actions, bound)
        self.actor_optim = torch.optim.Adam(self.actor_model.parameters(), lr=0.00001)

        self.critic_model = CriticNet(n_states, n_actions)
        self.critic_optim = torch.optim.Adam(self.critic_model.parameters(), lr=0.00002)

        # lr取一样的嘛？ 后期考虑

    def choose_action(self, s):
        s = torch.FloatTensor(s).to(device)
        mu, sigma = self.actor_model(s)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()
        punish = []
        for a__ in action:
            if a__ < 0:
                punish.append(a__,)
            if a__ > 1:
                punish.append(a__ - 1)
        punish = np.array(punish)
        action = torch.clamp(action, -s,  self.bound - s)
        return action, punish

    def discount_reward(self, rewards, s_):  # N步更新折扣奖励  rewards[batch] s_[num_divide * 2]
        s_ = torch.FloatTensor(s_).to(device)

        target = self.critic_model(s_).detach().to(device)  # torch.Size([1])
        target_list = []
        for r in rewards[::-1]:
            target = r + self.gamma * target
            target_list.append(target)
        target_list.reverse()
        target_list = torch.cat(target_list)  # torch.Size([batch])]
        return target_list

    def actor_learn(self, states, actions, advantage):  # [batch, num_d * 2] [batch, num_d * 2] [batch]
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)

        mu, sigma = self.actor_model(states)
        pi = torch.distributions.Normal(mu, sigma)

        old_mu, old_sigma = self.actor_old_model(states)
        old_pi = torch.distributions.Normal(old_mu, old_sigma)

        ratio = torch.exp(pi.log_prob(actions) - old_pi.log_prob(actions))  # [32,100]
        # ratio = pi.log_prob(actions) / (old_pi.log_prob(actions) + 0.000001)  # [32,100]

        surr = ratio * advantage.reshape(-1, 1)  # torch.Size([batch, 100]) * [100]
        loss = -torch.mean(
            torch.min(
                surr,
                torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage.reshape(-1, 1)))

        loss.to(device)

        self.actor_loss.append(loss)
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

    def critic_learn(self, states, targets):
        states = torch.FloatTensor(states).to(device)
        v = self.critic_model(states).reshape(1, -1).squeeze(0)

        loss_func = nn.MSELoss().to(device)
        loss = loss_func(v, targets)
        self.critic_loss.append(loss)

        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

    def cal_adv(self, states, targets):  # 计算优势函数
        states = torch.FloatTensor(states).to(device)
        v = self.critic_model(states)  # torch.Size([batch, num_divide * 2])
        advantage = targets - v.reshape(1, -1).squeeze(0)
        return advantage.detach()  # torch.Size([batch])

    def update(self, states, actions, targets):
        self.actor_old_model.load_state_dict(self.actor_model.state_dict())  # 首先更新旧模型
        advantage = self.cal_adv(states, targets)
        for i in range(self.a_update_steps):  # 更新多次
            self.actor_learn(states, actions, advantage)

        for i in range(self.c_update_steps):  # 更新多次
            self.critic_learn(states, targets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--len_episode', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.64)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--c_update_steps', type=int, default=10)
    parser.add_argument('--a_update_steps', type=int, default=10)
    args = parser.parse_args()

    env = final_mec_env.ENV
    # np.random.seed(1)
    num_device = 50
    num_mec = 4
    mec_max_cpu = 12
    mec_q_acc = 12
    t_threshold = 100
    offload_x = np.random.uniform(0, 1, num_device)  # 归一化
    local_f = np.random.uniform(0, 1, num_device)  # cpu归一化处理 [0,1] ->[2 * 10^9 , 4 * 10^9]
    # local_f = np.ceil(local_f * (10 ** 9))
    mec_f = np.array([10 ** 10, 10 ** 10, 10 ** 10, 10 ** 10])
    send_p = np.random.uniform(0, 1, num_device)  # 归一化
    channel_c = np.random.uniform(0, 1, num_device)  # 归一化
    d = np.random.randint(0., 500., (num_device, num_mec))
    b = np.random.uniform(100, 1000, num_device)
    b = np.ceil(b * 1000 * 1024)
    mec_c = np.random.uniform(0, 1, num_device)   # 归一化
    task_priority = np.random.uniform(0, 1, num_device)
    task_quality = np.random.uniform(0, 1, num_device)  # 归一化

    torch.manual_seed(args.seed)

    n_states = num_device * 4
    n_actions = num_device * 4
    bound = 1

    agent = PPO(n_states, n_actions, bound, args).to(device)

    all_ep_r = []
    all_cost_all = []
    all_delay_all = []
    all_energy_all = []
    all_s_all = []
    for episode in range(args.n_episodes):

        mec = env(num_device, offload_x, local_f, mec_f, send_p, channel_c, d, b, num_mec,
                            task_quality, mec_c, t_threshold, mec_max_cpu, mec_q_acc, task_priority)
        ep_r = 0
        s = np.append(mec.offload_x, mec.local_f)
        s = np.append(s, mec.send_p)
        # s = np.append(s, mec.channel_c)
        s = np.append(s, mec.mec_c)
        mec.calculate_normalized_parameter(offload_x, local_f, send_p, mec_c)
        states, actions, rewards = [], [], []
        for t in range(args.len_episode):

            a, punish = agent.choose_action(s)

            a = np.array(a)
            a = np.clip(a, -s, 1 - s)
            s, a, r, s_, e_all_, t_all_, cost_all_ = mec.step(s, a)
            ep_r += r
            states.append(s)
            actions.append(a)
            rewards.append(r)

            s = s_

            if (t + 1) % args.batch == 0 or t == args.len_episode - 1:  # N步更新
                states = np.array(states)
                actions = np.array(actions)
                rewards = np.array(rewards)

                targets = agent.discount_reward(rewards, s_)  # [detach]
                targets.to(device)
                agent.update(states, actions, targets)  # 进行actor和critic网络的更新
                states, actions, rewards = [], [], []

        print('Episode: ', episode, ' Reward: %.4f' % ep_r, 'delay:%.4f' % t_all_, 'consumption: %4f' % e_all_,
              'cost_all:%4f' % cost_all_)

        # if episode == 0:
        all_ep_r.append(ep_r)
        # else:
        #     all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)  # 平滑
        all_cost_all.append(cost_all_)
        all_delay_all.append(t_all_)
        all_energy_all.append(e_all_)
        all_s_all.append(s_)

    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.show()
    plt.plot(np.arange(len(all_cost_all)), all_cost_all)
    plt.show()
    plt.plot(np.arange(len(agent.actor_loss)), agent.actor_loss)
    plt.show()
    plt.plot(np.arange(len(agent.critic_loss)), agent.critic_loss)
    plt.show()


    print(np.min(all_delay_all))
    print(np.min(all_energy_all))
    print(np.min(all_cost_all))

    print(all_s_all[np.argmin(all_cost_all)][0:50])
    print(all_s_all[np.argmin(all_cost_all)][50:100])
    print(all_s_all[np.argmin(all_cost_all)][100:150])
    print(all_s_all[np.argmin(all_cost_all)][150:200])