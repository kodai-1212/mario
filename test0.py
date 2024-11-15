# 必要なライブラリのインポート
import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os
import copy

# Gymとマリオ環境のインポート
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

# ステージ一覧を作成
stage_names = [f'SuperMarioBros-{world}-{stage}-v0' for world in range(1, 9) for stage in range(1, 5)]

# 環境ラッパーの設定
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms_ = T.Compose([T.Resize(self.shape), T.Normalize(0, 255)])
        observation = transforms_(observation).squeeze(0)
        return observation
    

# DQNエージェントのクラス定義
class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        if h != 84 or w != 84:
            raise ValueError(f"Expected input height and width 84, but got {h}x{w}")
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.use_cuda = torch.cuda.is_available()
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.memory = deque(maxlen=100000)
        self.batch_size = 32
        self.gamma = 0.9
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.save_every = 5e5
        self.burnin = 1e4
        self.learn_every = 3
        self.sync_every = 1e4

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            state = state.__array__()
            state = torch.tensor(state).cuda() if self.use_cuda else torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        state = state.__array__()
        next_state = next_state.__array__()
        state = torch.tensor(state).cuda() if self.use_cuda else torch.tensor(state)
        next_state = torch.tensor(next_state).cuda() if self.use_cuda else torch.tensor(next_state)
        action = torch.tensor([action]).cuda() if self.use_cuda else torch.tensor([action])
        reward = torch.tensor([reward]).cuda() if self.use_cuda else torch.tensor([reward])
        done = torch.tensor([done]).cuda() if self.use_cuda else torch.tensor([done])
        self.memory.append((state, next_state, action, reward, done))

    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[np.arange(0, self.batch_size), action]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate), save_path)
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        if self.curr_step % self.save_every == 0:
            self.save()
        if self.curr_step < self.burnin or self.curr_step % self.learn_every != 0:
            return None, None
        state, next_state, action, reward, done = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)
        return (td_est.mean().item(), loss)
    
# 訓練の進捗を記録するクラス
class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Stage':>10}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )

    def log_episode(self, episode, step, stage, epsilon, ep_reward, ep_length, loss, q):
        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8}{step:8}{stage:10}{epsilon:10.3f}{ep_reward:15.3f}"
                f"{ep_length:15.3f}{loss:15.3f}{q:15.3f}\n"
            )


# 訓練プロセス
save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)
logger = MetricLogger(save_dir)

# マリオエージェントの初期化
mario = Mario(state_dim=(4, 84, 84), action_dim=2, save_dir=save_dir)  # 動作に合わせてaction_dimを設定

# ステージを順番に学習
for stage in stage_names:
    print(f"Starting training on stage: {stage}")
    # 環境の初期化
    env = gym_super_mario_bros.make(stage)
    env = JoypadSpace(env, [["right"], ["right", "A"]])  # 使用するアクションを定義
    # 環境にラッパーを適用
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)

    episodes = 10  # 各ステージでのエピソード数を指定
    stage_cleared = False
    for e in range(episodes):
        state = env.reset()
        ep_reward = 0
        ep_length = 0
        while True:
            action = mario.act(state)
            next_state, reward, done, info = env.step(action)
            mario.cache(state, next_state, action, reward, done)
            q, loss = mario.learn()

            # 訓練が始まるまで q と loss が None の場合があるため、記録する前にチェック
            if q is not None and loss is not None:
                ep_reward += reward
                ep_length += 1

            # 状態を更新
            state = next_state

            # エピソードが終了した場合にループを抜けて次のエピソードへ
            if done or info["flag_get"]:
                if info["flag_get"]:
                    print(f"Stage {stage} cleared in episode {e+1}!")
                    stage_cleared = True
                if q is not None and loss is not None:
                    logger.log_episode(e+1, mario.curr_step, stage, mario.exploration_rate, ep_reward, ep_length, loss, q)
                break
        # ステージをクリアしたら次のステージへ
        if stage_cleared:
            break


import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display

def display_animation(frames, interval=50):
    fig, ax = plt.subplots()
    ax.axis('off')
    patch = ax.imshow(frames[0], cmap='gray')

    def animate(i):
        patch.set_data(frames[i])

    ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=interval)
    plt.close(fig)
    return HTML(ani.to_jshtml())

def run_episode(env, agent):
    frames = []
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        state, _, done, info = env.step(action)
        # グレースケール画像として表示するために最初のフレームを取り出す
        frame = state[0].cpu().numpy()  # 形状が (84, 84) のデータ
        frames.append(frame)
        #if done or info["flag_get"]:
            #break
    return frames


# 最終ステージで訓練済みのエージェントで1エピソードを実行し、結果をアニメーションとして表示
env = gym_super_mario_bros.make(stage_names[-1])
env = JoypadSpace(env, [["right"], ["right", "A"]])
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

frames = run_episode(env, mario)
animation_html = display_animation(frames)
display(animation_html)