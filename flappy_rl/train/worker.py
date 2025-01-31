import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from model.a3c import ActorCritic

class Worker(mp.Process):
    def __init__(self, global_model, optimizer, rank, done_event,
                 max_episodes=10000, update_interval=32):
        super(Worker, self).__init__()
        self.rank = rank
        self.done_event = done_event
        self.max_episodes = max_episodes
        self.update_interval = update_interval
        
        # 创建本地模型
        self.local_model = ActorCritic().to('cpu')
        self.local_model.load_state_dict(global_model.state_dict())
        
        self.global_model = global_model
        self.optimizer = optimizer
        
        # 经验缓存
        self.states = []
        self.actions = []
        self.rewards = []
        self.policies = []
        
    def run(self):
        from game.environment import FlappyBirdEnv
        
        # 确保训练进程使用 rgb_array 模式
        env = FlappyBirdEnv(render_mode='rgb_array')
        
        try:
            episode = 0
            while episode < self.max_episodes and not self.done_event.is_set():
                state = env.reset()
                done = False
                score = 0
                
                # 收集经验
                while not done:
                    action, policy = self.local_model.get_action(state, 'cpu')
                    next_state, reward, done = env.step(action)
                    
                    self.states.append(state)
                    self.actions.append(action)
                    self.rewards.append(reward)
                    self.policies.append(policy)
                    
                    score += reward
                    state = next_state
                    
                    # 定期更新
                    if len(self.states) >= self.update_interval or done:
                        self.update_model(next_state, done, episode)
                        self.states.clear()
                        self.actions.clear()
                        self.rewards.clear()
                        self.policies.clear()
                
                episode += 1
                if episode % 10 == 0:
                    print(f"Worker {self.rank}, Episode {episode}, Score: {score}")
        finally:
            env.close()
    
    def update_model(self, next_state, done, episode):
        # 计算回报
        R = 0 if done else self.local_model(
            torch.FloatTensor(next_state).unsqueeze(0))[1].item()
        returns = []
        for reward in reversed(self.rewards):
            R = reward + 0.99 * R  # gamma = 0.99
            returns.insert(0, R)
            
        returns = torch.FloatTensor(returns).unsqueeze(1)
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        policies = torch.FloatTensor(self.policies)
        
        # 计算损失
        policies_now, values = self.local_model(states)
        advantages = returns - values.detach()
        
        # 使用PPO风格的损失计算
        ratio = torch.exp(torch.log(policies_now.gather(1, actions.unsqueeze(1))) - torch.log(torch.FloatTensor(policies).unsqueeze(1)))
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # 修正维度匹配
        critic_loss = F.smooth_l1_loss(values, returns)
        entropy_loss = -(policies_now * torch.log(policies_now + 1e-10)).sum(dim=1).mean()
        
        # 调整损失权重，增加critic loss的权重
        total_loss = actor_loss + 0.8 * critic_loss + 0.01 * entropy_loss
        if episode % 100 == 0:
            print(f"Worker {self.rank}, Episode {episode}, Actor Loss: {actor_loss.item()}, Critic Loss: {critic_loss.item()}, Entropy Loss: {entropy_loss.item()}")
        # 更新全局模型
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=0.5)
        
        for global_param, local_param in zip(self.global_model.parameters(),
                                           self.local_model.parameters()):
            global_param._grad = local_param.grad
        self.optimizer.step()
        
        # 同步本地模型
        self.local_model.load_state_dict(self.global_model.state_dict())