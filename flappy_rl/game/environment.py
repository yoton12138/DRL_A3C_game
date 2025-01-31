import pygame
import random
import numpy as np

class FlappyBirdEnv:
    def __init__(self, width=400, height=600, render_mode='rgb_array'):
        # 使用标志控制pygame初始化
        if not pygame.get_init():
            pygame.init()
        self.width = width
        self.height = height
        self.render_mode = render_mode
        
        # 创建显示窗口或Surface
        if render_mode == 'human':
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption('Flappy Bird RL')
        else:
            self.screen = pygame.Surface((width, height))
            
        # 添加时钟控制
        self.clock = pygame.time.Clock()
        self.FPS = 30
        
        # 初始化字体
        if not pygame.font.get_init():
            pygame.font.init()
        self.font = pygame.font.Font(None, 36)
        
        # 鸟的属性
        self.bird_size = 20
        self.bird_x = width // 4
        self.bird_y = height // 2
        self.bird_velocity = 0
        self.gravity = 0.8
        self.jump_velocity = -10
        
        # 管道属性
        self.pipe_width = 50
        self.pipe_gap = 150  # 固定的通过区域高度
        self.pipe_velocity = 3
        self.pipe_distance = 150  # 减小管道间距
        self.pipes = []
        
        # 游戏状态
        self.score = 0
        self.done = False
        
    def reset(self):
        """重置游戏状态"""
        self.bird_y = self.height // 2
        self.bird_velocity = 0
        self.pipes = []
        self.score = 0
        self.done = False
        self._add_pipe()
        return self._get_state()
    
    def step(self, action):
        """执行一步动作"""
        reward = 1  # 基础存活奖励
        
        # 处理动作，增加跳跃惩罚
        if action == 1:  # 跳跃
            self.bird_velocity = self.jump_velocity
            reward -= 0.5  # 每次跳跃的惩罚
        
        # 更新鸟的位置
        self.bird_velocity += self.gravity
        self.bird_y += self.bird_velocity
        
        # 获取当前最近的管道
        next_pipe = None
        for pipe in self.pipes:
            if pipe['x'] + self.pipe_width > self.bird_x:
                next_pipe = pipe
                break
        
        # 计算与最近管道的对齐奖励
        if next_pipe:
            pipe_center = next_pipe['gap_y'] + self.pipe_gap // 2
            height_diff = abs(self.bird_y - pipe_center)
            
            # 新的对齐奖励机制
            if height_diff < self.pipe_gap * 0.2:  # 当小鸟在管道中心20%范围内
                alignment_reward = 2.0  # 最大对齐奖励
            elif height_diff < self.pipe_gap * 0.4:  # 当小鸟在管道中心40%范围内
                alignment_reward = 1.0
            else:
                alignment_reward = 0.5 * np.exp(-height_diff / 30.0)  # 指数衰减奖励
            
            reward += alignment_reward
            
        
        # 边界惩罚
        if self.bird_y < self.height * 0.1 or self.bird_y > self.height * 0.9:
            reward -= 1  # 增加接近边界的惩罚
        
        # 检查碰撞
        if self._check_collision():
            if self.bird_y < 0 or self.bird_y > self.height:
                reward = -5  # 显著增加碰到上下边界的惩罚
            else:
                reward = -5  # 显著增加碰到管道的惩罚
            self.done = True
        
        # 检查得分
        for pipe in self.pipes:
            if not pipe['passed'] and pipe['x'] < self.bird_x:
                pipe['passed'] = True
                self.score += 1
                reward = 10  # 增加通过奖励
        
        # 更新管道位置
        for pipe in self.pipes:
            pipe['x'] -= self.pipe_velocity
        
        # 移除超出屏幕的管道
        self.pipes = [p for p in self.pipes if p['x'] > -self.pipe_width]
        
        # 添加新管道
        if len(self.pipes) == 0 or self.width - self.pipes[-1]['x'] >= self.pipe_distance:
            self._add_pipe()
        
        return self._get_state(), reward, self.done
    
    def _add_pipe(self):
        """添加新的管道，固定通过区域的高度"""
        # 将缺口中心位置限制在一定范围内
        min_gap_y = self.pipe_gap + 50  # 上边界留空
        max_gap_y = self.height - self.pipe_gap - 50  # 下边界留空
        gap_y = random.randint(min_gap_y, max_gap_y)
        
        self.pipes.append({
            'x': self.width,
            'gap_y': gap_y - self.pipe_gap // 2,  # 缺口底部位置
            'passed': False
        })
        
    def _check_collision(self):
        """检查碰撞"""
        if self.bird_y < 0 or self.bird_y > self.height:
            return True
            
        for pipe in self.pipes:
            if (self.bird_x + self.bird_size > pipe['x'] and 
                self.bird_x < pipe['x'] + self.pipe_width):
                if (self.bird_y < pipe['gap_y'] or 
                    self.bird_y + self.bird_size > pipe['gap_y'] + self.pipe_gap):
                    return True
        return False
    
    def _get_state(self):
        """获取游戏状态"""
        # 状态包括：鸟的高度、速度、最近管道的距离和高度差
        if not self.pipes:
            return np.array([0.5, 0, 1.0, 0])
            
        next_pipe = None
        for pipe in self.pipes:
            if pipe['x'] + self.pipe_width > self.bird_x:
                next_pipe = pipe
                break
                
        if next_pipe is None:
            return np.array([0.5, 0, 1.0, 0])
            
        # 归一化状态值
        bird_height = self.bird_y / self.height
        bird_vel = self.bird_velocity / 10.0
        pipe_dist = (next_pipe['x'] - self.bird_x) / self.width
        pipe_height_diff = (self.bird_y - (next_pipe['gap_y'] + self.pipe_gap/2)) / self.height
        
        return np.array([bird_height, bird_vel, pipe_dist, pipe_height_diff])
        
    def render(self):
        """渲染游戏画面"""
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return False
                    
            self.screen.fill((255, 255, 255))
            
            # 绘制鸟
            pygame.draw.rect(self.screen, (255, 0, 0),
                           (self.bird_x, self.bird_y, self.bird_size, self.bird_size))
            
            # 绘制管道
            for pipe in self.pipes:
                pygame.draw.rect(self.screen, (0, 255, 0),
                               (pipe['x'], 0, self.pipe_width, pipe['gap_y']))
                pygame.draw.rect(self.screen, (0, 255, 0),
                               (pipe['x'], pipe['gap_y'] + self.pipe_gap,
                                self.pipe_width, self.height))
            
            # 显示分数
            score_surface = self.font.render(f'Score: {self.score}', True, (0, 0, 0))
            self.screen.blit(score_surface, (10, 10))
            
            if self.render_mode == 'human':
                try:
                    pygame.display.flip()
                    self.clock.tick(self.FPS)
                except pygame.error:
                    return False
            
            return True
            
        except (pygame.error, Exception) as e:
            print(f"Render error: {e}")
            return False

    def close(self):
        if hasattr(self, 'screen'):
            pygame.quit() 