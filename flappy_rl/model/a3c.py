import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, input_dim=4, hidden_dims=[64, 64, 64]):
        super(ActorCritic, self).__init__()
        
        # 增加网络容量和深度
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.SiLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        
        # Actor网络 - 使用更小的初始化
        self.actor_layers = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[2] // 2),
            nn.SiLU(),
            nn.Linear(hidden_dims[2] // 2, 2),
            nn.Softmax(dim=-1)
        )
        
        # Critic网络
        self.critic_layers = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[2] // 2),
            nn.SiLU(),
            nn.Linear(hidden_dims[2] // 2, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 使用Kaiming初始化
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        shared_features = self.shared_layers(x)
        policy = self.actor_layers(shared_features)
        value = self.critic_layers(shared_features)
        return policy, value
    
    def get_action(self, state, device):
        state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            policy, _ = self(state)
            # 添加探索噪声，并确保概率值有效
            policy = policy + torch.randn_like(policy) * 0.05
            policy = torch.clamp(policy, min=1e-8)  # 确保没有负数
            policy = policy / policy.sum(dim=-1, keepdim=True)  # 重新归一化
            action = torch.multinomial(policy, 1).item()
        return action, policy[0][action].item()