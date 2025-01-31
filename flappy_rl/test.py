import torch
import time
from model.a3c import ActorCritic
from game.environment import FlappyBirdEnv

def test_model(model_path='flappy_bird_model.pth', num_episodes=5):
    # 测试时使用 human 模式
    env = FlappyBirdEnv(render_mode='human')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = ActorCritic().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败：{e}")
        return
    
    model.eval()
    
    try:
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_score = 0
            
            print(f"开始测试第 {episode + 1} 局")
            
            while not done:
                # 获取模型预测的动作
                action, _ = model.get_action(state, device)
                
                # 执行动作
                state, reward, done = env.step(action)
                episode_score += reward
                
                # 渲染游戏画面
                if not env.render():
                    print("游戏窗口被关闭")
                    return
                
                # 添加小延迟使游戏速度适中
                time.sleep(0.02)
            
            print(f"第 {episode + 1} 局得分: {env.score}")
            time.sleep(1)  # 局间暂停
            
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    finally:
        env.close()

if __name__ == "__main__":
    test_model() 