import torch
import torch.multiprocessing as mp
import threading
import time
from model.a3c import ActorCritic
from train.worker import Worker
from game.environment import FlappyBirdEnv
import os

def display_thread(global_model, done_event):
    """显示线程：实时展示训练效果"""
    try:
        env = FlappyBirdEnv(render_mode='human')
        device = torch.device("cpu")
        
        while not done_event.is_set():
            try:
                state = env.reset()
                done = False
                episode_score = 0
                
                while not done and not done_event.is_set():
                    with torch.no_grad():
                        action, _ = global_model.get_action(state, device)
                    
                    state, reward, done = env.step(action)
                    episode_score += reward
                    
                    # 渲染游戏画面
                    if not env.render():
                        print("显示窗口已关闭，停止显示线程...")
                        return
                    
                    time.sleep(0.02)
                
                print(f"Display Episode Score: {episode_score}")
                time.sleep(1)
                
            except Exception as e:
                print(f"显示线程错误: {e}")
                break
                
    except Exception as e:
        print(f"显示线程致命错误: {e}")
    finally:
        try:
            env.close()
        except:
            pass

def main():
    # 设置多进程方法
    if os.name == 'nt':  # Windows
        mp.set_start_method('spawn')
    
    # 创建全局模型
    global_model = ActorCritic().share_memory()
    optimizer = torch.optim.Adam(global_model.parameters(), lr=1e-3)
    
    # 减少工作进程数量
    num_workers = max(mp.cpu_count() // 4, 1)
    print(f"使用 {num_workers} 个训练进程")
    done_event = mp.Event()
    workers = []
    
    try:
        # 启动显示线程
        display_thread_obj = threading.Thread(
            target=display_thread,
            args=(global_model, done_event),
            daemon=True
        )
        display_thread_obj.start()
        
        # 启动训练进程
        for rank in range(num_workers):
            worker = Worker(global_model, optimizer, rank, done_event)
            worker.start()
            workers.append(worker)
            
        # 主线程等待用户输入
        while not done_event.is_set():
            time.sleep(1)
            if not display_thread_obj.is_alive():
                print("显示线程已停止，继续训练...")
                break
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        # 设置终止标志
        done_event.set()
        
        # 终止所有工作进程
        for worker in workers:
            worker.terminate()
            worker.join(timeout=1.0)
        
        # 等待显示线程结束
        if display_thread_obj.is_alive():
            display_thread_obj.join(timeout=1.0)
        
        # 保存模型
        try:
            torch.save(global_model.state_dict(), 'flappy_bird_model.pth')
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {e}")

if __name__ == '__main__':
    main() 