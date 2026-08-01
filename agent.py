import gymnasium as gym
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import flappy_bird_gymnasium
import torch
import yaml
import torch.nn as nn
import torch.optim as optim
import itertools
from dqn import DQN
from Experience_replay import ReplayMemory
import os
import argparse

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


class Agent:
    def __init__(self, params_set):
        self.params_set = params_set

        with open("parameters.yaml", "r") as f:
            all_params_set = yaml.safe_load(f)
            params = all_params_set[params_set]

            self.epsilon_init = params["epsilon_init"]
            self.epsilon_min = params["epsilon_min"]
            self.replay_memory_size = params["replay_memeory_size"]
            self.mini_batch_size = params["mini_batch_size"]
            self.network_sync_rate = params["network_sync_rate"]
            self.alpha = params["alpha"]
            self.gamma = params["gamma"]
            self.reward_threshold = params["reward_threshold"]
            self.learning_start = params.get("learning_start", 1000)

            self.loss_fnc = nn.SmoothL1Loss()
            self.optimizer = None
            self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.params_set}.log")
            self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.params_set}.pt")

    @staticmethod
    def get_state(env, obs, use_lidar_visual):
       
        if use_lidar_visual:
            obs, _ = env.unwrapped._get_observation_features()
        return obs

    def run(self, is_training=True, render=False):
       
        
        use_lidar_visual = (not is_training) and render

        env = gym.make(
            "FlappyBird-v0",
            render_mode="human" if render else None,
            use_lidar=use_lidar_visual,
        )
        num_states = 12 if use_lidar_visual else env.observation_space.shape[0] 
        num_action = env.action_space.n              
        print(f"Observation dim: {num_states}, Action dim: {num_action}")

        policy_dqn = DQN(num_states, num_action).to(device)

        best_reward = float("-inf")
        best_episode_reward = float("-inf")
        best_pipes = 0
        loss = 0.0
        epsilon = 0.0
        reward_history = deque(maxlen=100)
        pipes_history = deque(maxlen=100)

        start_episode = 0

        if is_training:
            writer = SummaryWriter(log_dir=f"runs/{self.params_set}_tensorboard")
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_action).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            total_steps = 0
            tau = 0.005   
            train_every = 4  
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)

           
            max_episodes = 10000
            min_lr = 1e-5
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max_episodes, eta_min=min_lr
            )

           
            epsilon_decay_episodes = 3000

            target_pipes = 15

           
            if os.path.exists(self.MODEL_FILE):
                checkpoint = torch.load(self.MODEL_FILE, map_location=device)
                policy_dqn.load_state_dict(checkpoint["model"])
                target_dqn.load_state_dict(checkpoint["model"])
                if "optimizer" in checkpoint:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                epsilon = checkpoint.get("epsilon", self.epsilon_init)
                best_reward = checkpoint.get("best_reward", float("-inf"))
                best_pipes = checkpoint.get("best_pipes", 0)
                start_episode = checkpoint.get("episode", -1) + 1

                
                for _ in range(start_episode):
                    scheduler.step()

                print(f"Resuming training from episode {start_episode} "
                      f"(best_reward={best_reward:.2f}, best_pipes={best_pipes}, "
                      f"epsilon={epsilon:.4f}, "
                      f"lr={self.optimizer.param_groups[0]['lr']:.6f})")
            else:
                print("No existing checkpoint found — starting training from scratch.")
        else:
            if not os.path.exists(self.MODEL_FILE):
                print(f"ERROR: No model file found at '{self.MODEL_FILE}'.")
                print("Please train first: python agent.py flappybirdv0 --train")
                return
            checkpoint = torch.load(self.MODEL_FILE, map_location=device)
            try:
                policy_dqn.load_state_dict(checkpoint["model"])
            except RuntimeError as e:
                print("ERROR: Saved model is incompatible with current network architecture.")
                print("Delete old models and retrain: del runs\\*.pt")
                print("Then: python agent.py flappybirdv0 --train")
                print(f"\nDetails: {e}")
                return
            policy_dqn.eval()

        for episode in itertools.count(start=start_episode):
            state, _ = env.reset()
            state = self.get_state(env, state, use_lidar_visual)
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            episode_reward = 0.0
            pipes_crossed = 0
            prev_score = 0

            while not terminated and episode_reward < self.reward_threshold:
               
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.long, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                next_state, reward, terminated, _, info = env.step(action.item())

                episode_reward += reward

               
                current_score = info.get('score', 0)
                if current_score > prev_score:
                    pipes_crossed = current_score
                    prev_score = current_score

                next_state = self.get_state(env, next_state, use_lidar_visual)
                next_state = torch.tensor(next_state, dtype=torch.float, device=device)

               
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, next_state, reward, terminated))
                    total_steps += 1

                   
                    if len(memory) >= self.learning_start and total_steps % train_every == 0:
                        mini_batch = memory.sample(self.mini_batch_size)
                        loss = self.optimize(mini_batch, policy_dqn, target_dqn)

                       
                        for target_param, policy_param in zip(target_dqn.parameters(), policy_dqn.parameters()):
                            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

                state = next_state

            reward_history.append(episode_reward)
            pipes_history.append(pipes_crossed)
            rolling_avg = sum(reward_history) / len(reward_history)
            avg_pipes = sum(pipes_history) / len(pipes_history)

            if is_training:
                
                if total_steps > self.learning_start:
                    scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']

                print(f"episode {episode+1} | reward: {episode_reward:.2f} | "
                      f"pipes: {pipes_crossed} | avg(100): {rolling_avg:.2f} | "
                      f"avg_pipes(100): {avg_pipes:.1f} | "
                      f"epsilon: {epsilon:.4f} | lr: {current_lr:.6f}")

                writer.add_scalar("Reward/episode", episode_reward, episode)
                writer.add_scalar("Reward/rolling_avg_100", rolling_avg, episode)
                writer.add_scalar("Pipes/episode", pipes_crossed, episode)
                writer.add_scalar("Pipes/avg_100", avg_pipes, episode)
                writer.add_scalar("Loss", loss, total_steps)
                writer.add_scalar("Epsilon", epsilon, episode)
                writer.add_scalar("LearningRate", current_lr, episode)

               
                epsilon = max(
                    self.epsilon_init - (self.epsilon_init - self.epsilon_min) * (episode / epsilon_decay_episodes),
                    self.epsilon_min
                )

                
                def save_checkpoint(path):
                    torch.save({
                        "model": policy_dqn.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "episode": episode,
                        "epsilon": epsilon,
                        "best_reward": best_reward,
                        "best_pipes": best_pipes,
                    }, path)

               
                if len(reward_history) >= 100 and rolling_avg > best_reward:
                    log_msg = f"new best avg(100) = {rolling_avg:.2f} at episode {episode+1}"
                    print(f"  >>> {log_msg}")
                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")
                    save_checkpoint(self.MODEL_FILE)
                    best_reward = rolling_avg

                if episode_reward > best_episode_reward:
                    best_episode_reward = episode_reward
                    best_ep_model = os.path.join(RUNS_DIR, f"{self.params_set}_best_episode.pt")
                    save_checkpoint(best_ep_model)
                    log_msg = f"new best episode reward = {episode_reward:.2f} (pipes: {pipes_crossed}) at episode {episode+1}"
                    print(f"  >>> {log_msg}")
                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")

                if pipes_crossed > best_pipes:
                    best_pipes = pipes_crossed
                    best_pipes_model = os.path.join(RUNS_DIR, f"{self.params_set}_best_pipes.pt")
                    save_checkpoint(best_pipes_model)
                    log_msg = f"NEW PIPES RECORD = {pipes_crossed} pipes (reward: {episode_reward:.2f}) at episode {episode+1}"
                    print(f"\n{'='*60}")
                    print(f"  {log_msg}")
                    print(f"{'='*60}\n")
                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")

                if len(pipes_history) >= 100 and avg_pipes >= target_pipes:
                    print(f"\n{'='*60}")
                    print(f"  TARGET REACHED! avg_pipes(100) = {avg_pipes:.1f} >= {target_pipes}")
                    print(f"  Saving final model and stopping training.")
                    print(f"{'='*60}\n")
                    save_checkpoint(self.MODEL_FILE)
                    break

                if episode >= max_episodes:
                    print(f"\nReached max episodes ({max_episodes}). Stopping.")
                    save_checkpoint(self.MODEL_FILE)
                    break

            else:
                if episode_reward > best_episode_reward:
                    best_episode_reward = episode_reward
                    print(f"\n{'='*60}")
                    print(f"  NEW BEST! episode {episode+1} | reward: {episode_reward:.2f} | pipes: {pipes_crossed}")
                    print(f"{'='*60}\n")

        if is_training:
            writer.close()
        env.close()

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, next_states, rewards, terminations = zip(*mini_batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            best_next_actions = policy_dqn(next_states).argmax(dim=1)
            target_q = rewards + self.gamma * (1 - terminations) * \
                target_dqn(next_states).gather(dim=1, index=best_next_actions.unsqueeze(dim=1)).squeeze()

        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        loss = self.loss_fnc(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Test Model")
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(params_set=args.hyperparameters)
    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)