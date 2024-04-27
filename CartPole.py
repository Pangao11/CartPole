import argparse
import datetime
import time
from collections import deque

from torch.distributions import Bernoulli
from torch.autograd import Variable
import gym
from torch import nn

# 这里需要改成自己的RL_Utils.py文件的路径
from RL_Utils import *


class MemoryQueue:
    def __init__(self):
        self.buffer = deque()

    def push(self, transitions):
        self.buffer.append(transitions)

    def sample(self):
        batch = list(self.buffer)
        return zip(*batch)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


# 策略网络（全连接网络）
class DNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        """ 初始化策略网络，为全连接网络
            input_dim: 输入的特征数即环境的状态维度
            output_dim: 输出的动作维度
        """
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # 输出层

    def forward(self, x):
        # 各层对应的激活函数
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


# PolicyGradient智能体对象
class PolicyGradient:
    def __init__(self, model, memory, arg_dict):
        # 未来奖励衰减因子
        self.gamma = arg_dict['gamma']
        self.device = torch.device(arg_dict['device'])
        self.memory = memory
        # 策略网络
        self.policy_net = model.to(self.device)
        # 优化器
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=arg_dict['lr'])

    def sample_action(self, state):
        state = torch.from_numpy(state).float()
        state = Variable(state)
        probs = self.policy_net(state.to(self.device))
        m = Bernoulli(probs)  # 伯努利分布
        action = m.sample()
        action = int(action.item())  # 转为标量
        return action

    def predict_action(self, state):

        state = torch.from_numpy(state).float()
        state = Variable(state)
        probs = self.policy_net(state.to(self.device))
        m = Bernoulli(probs)  # 伯努利分布
        action = m.sample()
        action = int(action.item())  # 转为标量
        return action

    def update(self):
        state_pool, action_pool, reward_pool = self.memory.sample()
        state_pool, action_pool, reward_pool = list(state_pool), list(action_pool), list(reward_pool)
        # 对奖励进行修正，考虑未来，并加入衰减因子
        running_add = 0
        for i in reversed(range(len(reward_pool))):
            if reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * self.gamma + reward_pool[i]
                reward_pool[i] = running_add

        reward_mean = np.mean(reward_pool)  # 求奖励均值
        reward_std = np.std(reward_pool)  # 求奖励标准差
        for i in range(len(reward_pool)):
            # 标准化奖励
            reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

        # 梯度下降
        self.optimizer.zero_grad()
        for i in range(len(reward_pool)):
            state = state_pool[i]
            action = Variable(torch.FloatTensor([action_pool[i]]))
            reward = reward_pool[i]
            state = Variable(torch.from_numpy(state).float())
            probs = self.policy_net(state.to(self.device))
            m = Bernoulli(probs)
            # 加权(reward)损失函数，加负号(将最大化问题转化为最小化问题)
            loss = -m.log_prob(action.to(self.device)) * reward
            loss.backward()
        self.optimizer.step()
        self.memory.clear()

    def save_model(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), path + 'checkpoint.pt')

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path + 'checkpoint.pt'))


# 训练函数
def train(arg_dict, env, agent):
    # 开始计时
    startTime = time.time()
    print(f"环境名: {arg_dict['env_name']}, 算法名: {arg_dict['algo_name']}, Device: {arg_dict['device']}")
    print("开始训练智能体......")
    # 记录每个epoch的奖励
    rewards = []
    for epoch in range(arg_dict['train_eps']):
        state = env.reset()
        ep_reward = 0
        for _ in range(arg_dict['ep_max_steps']):
            # 画图
            if arg_dict['train_render']:
                env.render()
            # 采样
            action = agent.sample_action(state)
            # 执行动作，获取下一个状态、奖励和结束状态
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            # 如果回合结束，则奖励为0
            if done:
                reward = 0
            # 讲采样的数据存起来
            agent.memory.push((state, float(action), reward))
            # 更新状态：当前状态等于下一个状态
            state = next_state
            # 如果回合结束，则跳出循环
            if done:
                break
        if (epoch + 1) % 10 == 0:
            print(f"Epochs：{epoch + 1}/{arg_dict['train_eps']}, Reward:{ep_reward:.2f}")
        # 每采样几个回合就对智能体做一次更新
        if (epoch + 1) % arg_dict['update_fre'] == 0:
            agent.update()
        rewards.append(ep_reward)
    print('训练结束 , 用时: ' + str(time.time() - startTime) + " s")
    # 关闭环境
    env.close()
    return {'episodes': range(len(rewards)), 'rewards': rewards}


# 测试函数
def test(arg_dict, env, agent):
    startTime = time.time()
    print("开始测试智能体......")
    print(f"环境名: {arg_dict['env_name']}, 算法名: {arg_dict['algo_name']}, Device: {arg_dict['device']}")
    # 记录每个epoch的奖励
    rewards = []
    for epoch in range(arg_dict['test_eps']):
        state = env.reset()
        ep_reward = 0
        for _ in range(arg_dict['ep_max_steps']):
            # 画图
            if arg_dict['test_render']:
                env.render()
            action = agent.predict_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                reward = 0
            state = next_state
            if done:
                break
        print(f"Epochs: {epoch + 1}/{arg_dict['test_eps']}，Reward: {ep_reward:.2f}")
        rewards.append(ep_reward)
    print("测试结束 , 用时: " + str(time.time() - startTime) + " s")
    env.close()
    return {'episodes': range(len(rewards)), 'rewards': rewards}


# 创建环境和智能体
def create_env_agent(arg_dict):
    # 创建环境
    env = gym.make(arg_dict['env_name'])
    # 设置随机种子
    all_seed(env, seed=arg_dict["seed"])
    # 获取状态数
    try:
        n_states = env.observation_space.n
    except AttributeError:
        n_states = env.observation_space.shape[0]
    # 获取动作数
    n_actions = env.action_space.n
    print(f"状态数: {n_states}, 动作数: {n_actions}")
    # 将状态数和动作数加入算法参数字典
    arg_dict.update({"n_states": n_states, "n_actions": n_actions})
    model = DNN(n_states, 1, hidden_dim=arg_dict['hidden_dim'])
    memory = MemoryQueue()
    # 实例化智能体对象
    agent = PolicyGradient(model, memory, arg_dict)
    # 返回环境，智能体
    return env, agent


if __name__ == '__main__':
    # 防止报错 OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # 获取当前路径
    curr_path = os.path.dirname(os.path.abspath(__file__))
    # 获取当前时间
    curr_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    # 相关参数设置
    parser = argparse.ArgumentParser(description="hyper parameters")
    parser.add_argument('--algo_name', default='PolicyGradient', type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='CartPole-v0', type=str, help="name of environment")
    parser.add_argument('--train_eps', default=1000, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=20, type=int, help="episodes of testing")
    parser.add_argument('--ep_max_steps', default=100000, type=int,
                        help="steps per episode, much larger value can simulate infinite steps")
    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser.add_argument('--lr', default=0.001, type=float, help="learning rate")
    parser.add_argument('--update_fre', default=10, type=int)
    parser.add_argument('--hidden_dim', default=36, type=int)
    parser.add_argument('--device', default='cpu', type=str, help="cpu or cuda")
    parser.add_argument('--seed', default=520, type=int, help="seed")
    parser.add_argument('--show_fig', default=False, type=bool, help="if show figure or not")
    parser.add_argument('--save_fig', default=True, type=bool, help="if save figure or not")
    parser.add_argument('--train_render', default=False, type=bool,
                        help="Whether to render the environment during training")
    parser.add_argument('--test_render', default=True, type=bool,
                        help="Whether to render the environment during testing")
    args = parser.parse_args()
    default_args = {'result_path': f"{curr_path}/outputs/{args.env_name}/{curr_time}/results/",
                    'model_path': f"{curr_path}/outputs/{args.env_name}/{curr_time}/models/",
                    }
    # 将参数转化为字典 type(dict)
    arg_dict = {**vars(args), **default_args}
    print("算法参数字典:", arg_dict)

    # 创建环境和智能体
    env, agent = create_env_agent(arg_dict)
    # 传入算法参数、环境、智能体，然后开始训练
    res_dic = train(arg_dict, env, agent)
    print("算法返回结果字典:", res_dic)
    # 保存相关信息
    agent.save_model(path=arg_dict['model_path'])
    save_args(arg_dict, path=arg_dict['result_path'])
    save_results(res_dic, tag='train', path=arg_dict['result_path'])
    plot_rewards(res_dic['rewards'], arg_dict, path=arg_dict['result_path'], tag="train")

    # =================================================================================================
    # 创建新环境和智能体用来测试
    print("=" * 300)
    env, agent = create_env_agent(arg_dict)
    # 加载已保存的智能体
    agent.load_model(path=arg_dict['model_path'])
    res_dic = test(arg_dict, env, agent)
    save_results(res_dic, tag='test', path=arg_dict['result_path'])
    plot_rewards(res_dic['rewards'], arg_dict, path=arg_dict['result_path'], tag="test")
