import gym
import gym
import numpy as np
import torch
import torch.optim as optim
import wandb
from gym import Wrapper

from config import initial_exploration, batch_size, update_target, goal_score, log_interval, device, \
    replay_memory_capacity, lr, local_mini_batch
from embedding_model import EmbeddingModel, compute_intrinsic_reward
from memory import Memory, LocalBuffer
from model import R2D2


def get_action(state, target_net, epsilon, env, hidden):
    action, hidden = target_net.get_action(state, hidden)

    if np.random.rand() <= epsilon:
        return env.action_space.sample(), hidden
    else:
        return action, hidden


def update_target_model(online_net, target_net):
    target_net.load_state_dict(online_net.state_dict())


class Maze(Wrapper):

    def step(self, action: int):
        obs, rew, done, info = super().step(['N', 'E', 'S', 'W'][action])
        self.set.add((obs[0], obs[1]))
        if rew > 0:
            rew = 10
        return obs/10, rew, done, info


    def reset(self):
        self.set = set()
        return super().reset()



def main():
    import gym_maze
    # env = Maze(gym.make("maze-sample-5x5-v0"))
    env = Maze(gym.make("maze-sample-10x10-v0"))

    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)
    env.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)

    epsilon_decay = 0.000003
    wandb.init(project="ngu-maze", config={
        "RANDOM_SEED": RANDOM_SEED,
        "epsilon_decay": epsilon_decay
    })

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    online_net = R2D2(num_inputs, num_actions)
    target_net = R2D2(num_inputs, num_actions)
    update_target_model(online_net, target_net)
    embedding_model = EmbeddingModel(obs_size=num_inputs, num_outputs=num_actions)
    embedding_loss = 0

    optimizer = optim.Adam(online_net.parameters(), lr=lr)

    online_net.to(device)
    target_net.to(device)
    online_net.train()
    target_net.train()
    memory = Memory(replay_memory_capacity)
    epsilon = 1.0
    steps = 0
    loss = 0
    local_buffer = LocalBuffer()
    sum_reward = 0
    sum_augmented_reward = 0
    sum_obs_set = 0

    for episode in range(30000):
        done = False
        state = env.reset()
        state = torch.Tensor(state).to(device)

        hidden = (torch.Tensor().new_zeros(1, 1, 16), torch.Tensor().new_zeros(1, 1, 16))

        episodic_memory = [embedding_model.embedding(state)]

        episode_steps = 0
        horizon = 200  # TODO: horizon 100 for map 5x5
        while not done:
            steps += 1
            episode_steps += 1

            action, new_hidden = get_action(state, target_net, epsilon, env, hidden)

            next_state, env_reward, done, _ = env.step(action)

            if env_reward > 0:
                print("Goal!", env_reward, done, episode_steps)
            # env.render()

            next_state = torch.Tensor(next_state)

            next_state_emb = embedding_model.embedding(next_state)
            intrinsic_reward = compute_intrinsic_reward(episodic_memory, next_state_emb)
            episodic_memory.append(next_state_emb)
            beta = 0.0001
            augmented_reward = env_reward + beta * intrinsic_reward

            mask = 0 if done else 1

            local_buffer.push(state, next_state, action, augmented_reward, mask, hidden)
            hidden = new_hidden
            if len(local_buffer.memory) == local_mini_batch:
                batch, lengths = local_buffer.sample()
                td_error = R2D2.get_td_error(online_net, target_net, batch, lengths)
                memory.push(td_error, batch, lengths)

            sum_reward += env_reward
            state = next_state
            sum_augmented_reward += augmented_reward

            if steps > initial_exploration and len(memory) > batch_size:
                # epsilon -= 0.00001 # episolon pour 5x5
                epsilon -= epsilon_decay # episolon pour 10x10 try
                epsilon = max(epsilon, 0.4)

                batch, indexes, lengths = memory.sample(batch_size)
                loss, td_error = R2D2.train_model(online_net, target_net, optimizer, batch, lengths)

                # NGU TODO: enable embedding training
                # embedding_loss = embedding_model.train_model(batch)

                memory.update_priority(indexes, td_error, lengths)

                if steps % update_target == 0:
                    # print("UPDATE TARGET NET", steps, update_target, steps % update_target)
                    update_target_model(online_net, target_net)

            if episode_steps >= horizon or done:
                sum_obs_set += len(env.set)
                break

        if episode > 0 and episode % log_interval == 0:
            mean_reward = sum_reward/log_interval
            mean_augmented_reward = sum_augmented_reward / log_interval
            metrics = {
                    'episode': episode,
                    'mean_reward': mean_reward,
                    'epsilon': epsilon,
                    'embedding_loss': embedding_loss,
                    'loss': loss,
                    'mean_augmented_reward': mean_augmented_reward,
                    'steps': steps,
                    'sum_obs_set': sum_obs_set/log_interval,
                 }
            print(metrics)
            wandb.log(metrics)

            sum_reward = 0
            sum_augmented_reward = 0
            sum_obs_set = 0

            if mean_reward > goal_score:
                break


if __name__ == "__main__":
    main()
