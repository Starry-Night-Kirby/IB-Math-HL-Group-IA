import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        if n_states == 0:
            print("[WARNING] generate_batches() called with empty memory")
            return (np.array([]), np.array([]), np.array([]),
                    np.array([]), np.array([]), np.array([]), [])
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batch_starts = np.arange(0, n_states, self.batch_size)
        batches = [indices[i:i+self.batch_size] for i in batch_starts]
        return (np.array(self.states), np.array(self.actions), np.array(self.probs),
                np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches)

    def store_memory(self, state, action, prob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states, self.actions, self.probs, self.vals, self.rewards, self.dones = ([], [], [], [], [], [])


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir="tmp/ppo"):
        super(ActorNetwork, self).__init__()
        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Tanh()
        )
        # action bounds
        self.action_low = T.tensor([-4.0, -4.0, 0.0], dtype=T.float32)
        self.action_high = T.tensor([4.0, 4.0, 8.99], dtype=T.float32)
        # learnable log std
        self.log_std = nn.Parameter(T.zeros(n_actions))
        # softplus to ensure positive std
        self.softplus = nn.Softplus()
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.actor(state)
        if T.isnan(x).any():
            print("[ERROR] Actor forward() produced NaNs in output!")
            print("Input state:", state)
            print("Output logits:", x)
            raise ValueError("NaNs in actor output!")
        std = self.softplus(self.log_std).expand_as(x)
        return Normal(x, std)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir="tmp/ppo"):
        super(CriticNetwork, self).__init__()
        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0001, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, prob, val, reward, done):
        self.memory.store_memory(state, action, prob, val, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        # convert to tensor
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation, dtype=np.float32)
        obs_tensor = T.tensor(observation, dtype=T.float32).unsqueeze(0).to(self.actor.device)

        # get distribution and value
        dist = self.actor(obs_tensor)
        value = self.critic(obs_tensor)

        # sample and compute log-prob
        raw_action = dist.sample().squeeze(0)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)

        # send back numpy action; env.step will do clipping/modulus
        action = raw_action.detach().cpu().numpy()
        return action, log_prob.item(), value.item()

    def learn(self):
        if len(self.memory.states) == 0:
            print("[WARNING] Skipping learn() - memory has no data")
            return
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()
            # compute advantages
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k+1] * (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage, dtype=T.float32).to(self.actor.device)
            values = T.tensor(values, dtype=T.float32).to(self.actor.device)

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float32).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch], dtype=T.float32).to(self.actor.device).squeeze()
                actions = T.tensor(action_arr[batch], dtype=T.float32).to(self.actor.device)
                # debug NaNs
                if T.isnan(states).any():
                    print("[ERROR] NaN detected in state tensor")
                    return
                dist = self.actor(states)
                critic_value = self.critic(states).squeeze()
                new_probs = dist.log_prob(actions).sum(dim=-1)
                prob_ratio = (new_probs.exp() / old_probs.exp())
                # actor loss
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped).mean()
                # critic loss
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()
                # total loss
                total_loss = actor_loss + 0.5 * critic_loss
                # backprop
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                # gradient clipping
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        self.memory.clear_memory()

# import os
# import numpy as np
# import torch as T
# import torch.nn as nn
# import torch.optim as optim
# from torch.distributions.categorical import Categorical

# class PPOMemory:
#     def __init__(self, batch_size):
#         self.states = []
#         self.probs = []
#         self.vals = []
#         self.actions = []
#         self.rewards = []
#         self.dones = []

#         self.batch_size = batch_size

#     def generate_batches(self):
#         n_states = len(self.states)
#         if n_states == 0:
#             print("[WARNING] generate_batches() called with empty memory")
#             return(
#                 np.array([]),
#                 np.array([]),
#                 np.array([]),
#                 np.array([]),
#                 np.array([]),
#                 np.array([]),
#                 []
#             )


#         batch_start = np.arange(0,n_states,self.batch_size)
#         indices = np.arange(n_states,dtype = np.int64)
#         np.random.shuffle(indices)
#         batches = [indices[i:i+self.batch_size] for i in batch_start]

#         return np.array(self.states),\
#                 np.array(self.actions),\
#                 np.array(self.probs),\
#                 np.array(self.vals),\
#                 np.array(self.rewards),\
#                 np.array(self.dones),\
#                 batches
    
#     def store_memory(self,state,action,probs,vals,rewards,dones):
#         self.states.append(state)
#         self.probs.append(probs)
#         self.actions.append(action)
#         self.vals.append(vals)
#         self.rewards.append(rewards)
#         self.dones.append(dones)

#     def clear_memory(self):
#         self.states = []
#         self.probs = []
#         self.vals = []
#         self.actions = []
#         self.rewards = []
#         self.dones = []


# class ActorNetwork (nn.Module):
#     def __init__(self,n_actions,input_dims,alpha,fc1_dims=256,fc2_dims=256,chkpt_dir = "tmp/ppo"):
#         super(ActorNetwork,self).__init__()
#         os.makedirs(chkpt_dir, exist_ok=True)
#         self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
#         self.actor = nn.Sequential(
#             nn.Linear(*input_dims, fc1_dims),
#             nn.ReLU(),
#             nn.Linear(fc1_dims,fc2_dims),
#             nn.ReLU(),
#             nn.Linear(fc2_dims, n_actions)

#             )
#         self.action_low = np.array([-4.0, -4.0, 0.0])
#         self.action_high = np.array([4.0, 4.0, 8.99])
#         self.log_std = nn.Parameter(T.zeros(n_actions))
#         self.optimizer = optim.Adam(self.parameters(),lr = alpha)
#         self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
#         self.to(self.device)

# def forward(self, state):
#     x = self.actor(state)

#     if T.isnan(x).any():
#         print("[ERROR] Actor forward() produced NaNs in output!")
#         print("Input state:", state)
#         print("Output logits:", x)
#         raise ValueError("NaNs in actor output!")

#     std = self.std.expand_as(x)
#     dist = T.distributions.Normal(x, std)
#     return dist
    
#     def save_checkpoint(self):
#         T.save(self.state_dict(),self.checkpoint_file)

#     def load_checkpoint(self):
#         self.load_state_dict(T.load(self.checkpoint_file))


# class CriticNetwork(nn.Module):
#     def __init__(self, input_dims,alpha, fc1_dims = 256, fc2_dims = 256, chkpt_dir = "tmp/ppo"):
#         super(CriticNetwork, self).__init__()
#         os.makedirs(chkpt_dir, exist_ok=True)
#         self.checkpoint_file = os.path.join(chkpt_dir, "critic_torch_ppo")
#         self.critic = nn.Sequential(
#             nn.Linear(*input_dims, fc1_dims),
#             nn.ReLU(),
#             nn.Linear(fc1_dims,fc2_dims),
#             nn.ReLU(),
#             nn.Linear(fc2_dims, 1)
#         )
#         self.optimizer = optim.Adam(self.parameters(),lr = alpha)
#         self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
#         self.to(self.device)

#     def forward(self,state):
#         value = self.critic(state)
#         return value

    
#     def save_checkpoint(self):
#         T.save(self.state_dict(),self.checkpoint_file)

#     def load_checkpoint(self):
#         self.load_state_dict(T.load(self.checkpoint_file))

# class Agent:
#     def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
#             policy_clip=0.2, batch_size=64, n_epochs=10):
#         self.gamma = gamma
#         self.policy_clip = policy_clip
#         self.n_epochs = n_epochs
#         self.gae_lambda = gae_lambda

#         self.actor = ActorNetwork(n_actions, input_dims, alpha)
#         self.critic = CriticNetwork(input_dims, alpha)
#         self.memory = PPOMemory(batch_size)
       
#     def remember(self, state, action, probs, vals, reward, done):
#         self.memory.store_memory(state, action, probs, vals, reward, done)

#     def save_models(self):
#         print('... saving models ...')
#         self.actor.save_checkpoint()
#         self.critic.save_checkpoint()

#     def load_models(self):
#         print('... loading models ...')
#         self.actor.load_checkpoint()
#         self.critic.load_checkpoint()

#     def choose_action(self, observation):
#         # print(f"[DEBUG] choose_action() called with obs: {observation}")
#         if observation is None:
#             raise ValueError("Observation is None")
#         if not isinstance(observation, np.ndarray):
#             observation = np.array(observation, dtype=np.float32)
#         if observation.shape != (5,):
#             raise ValueError(f"Observation must have shape (5,), got {observation.shape}")

    
#         obs_tensor = T.tensor(observation, dtype=T.float32).unsqueeze(0).to(self.actor.device)

#         dist = self.actor(obs_tensor)
#         value = self.critic(obs_tensor)

    
#         action = dist.sample()
#         action = T.clamp(action, T.tensor(self.actor.action_low).to(self.actor.device), T.tensor(self.actor.action_high).to(self.actor.device))

#         log_prob = dist.log_prob(action).sum(dim=-1)

    

#         action_np = action.squeeze(0).detach().cpu().numpy()

#         # Defensive conversion to 1D array

#         if isinstance(action_np, np.ndarray):
#             if action_np.ndim == 0:
#                 action_np = np.array([action_np], dtype=np.float32)
#             elif action_np.ndim > 1:
#                 action_np = action_np.squeeze()
#         else:
#             action_np = np.array([action_np], dtype=np.float32)

#         # print(f"[RETURN] action_np: {action_np},shape : {action_np.shape}")
#         # print(f"[RETURN] log_prob {log_prob.item()},value : {value.item()}")


#         return action_np, log_prob.item(), value.item()

#     # def choose_action(self, observation):
#     #     if not isinstance(observation, np.ndarray):
#     #         observation = np.array(observation,dtype=np.float32)
#     #     if observation.shape != (5,):
#     #         raise ValueError(f"[ERROR] Observation must be shape (5,), got: {observation.shape}")
#     #     state = T.tensor(observation, dtype=T.float).unsqueeze(0).to(self.actor.device)

#     #     dist = self.actor(state)
#     #     value = self.critic(state)
#     #     action = dist.sample()
#     #     log_prob = dist.log_prob(action)

#     #     # probs = T.squeeze(dist.log_prob(action)).item()
#     #     # action = T.squeeze(action).item()
#     #     # value = T.squeeze(value).item()

#     #     action = action.detach().cpu().numpy
#     #     if isinstance(action,np.ndarray):
#     #         if action.ndim == 0:
#     #             action_np = np.array([action],dtype=np.float32)
#     #         elif action_np.ndim > 1:
#     #             action_np = action.squeeze()

#     #         return action_np, log_prob.item(), value.item()

#     def learn(self):
#         if len(self.memory.states)==0:
#             print("[WARNING] Skipping learn() - memory has no data")
#             return

#         for _ in range(self.n_epochs):
#             state_arr, action_arr, old_prob_arr, vals_arr,\
#             reward_arr, dones_arr, batches = \
#                     self.memory.generate_batches()

#             values = vals_arr
#             advantage = np.zeros(len(reward_arr), dtype=np.float32)

#             for t in range(len(reward_arr)-1):
#                 discount = 1
#                 a_t = 0
#                 for k in range(t, len(reward_arr)-1):
#                     a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
#                             (1-int(dones_arr[k])) - values[k])
#                     discount *= self.gamma*self.gae_lambda
#                 advantage[t] = a_t
#             advantage = T.tensor(advantage).to(self.actor.device)

#             values = T.tensor(values).to(self.actor.device)
#             for batch in batches:
#                 states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
#                 old_probs = T.tensor(old_prob_arr[batch], dtype=T.float32).to(self.actor.device).squeeze()
#                 actions = T.tensor(action_arr[batch]).to(self.actor.device)

#                 if T.isnan(states).any():
#                     print("[ERROR] NaN detected in state tensor")
#                     print(states)
#                     return
#                 dist = self.actor(states)
#                 critic_value = self.critic(states)

#                 critic_value = T.squeeze(critic_value)

#                 new_probs = dist.log_prob(actions).sum(axis=-1)
#                 prob_ratio = new_probs.exp() / old_probs.exp()
#                 #prob_ratio = (new_probs - old_probs).exp()
#                 weighted_probs = advantage[batch] * prob_ratio
#                 weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
#                         1+self.policy_clip)*advantage[batch]
#                 actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

#                 returns = advantage[batch] + values[batch]
#                 critic_loss = (returns-critic_value)**2
#                 critic_loss = critic_loss.mean()

#                 total_loss = actor_loss + 0.5*critic_loss
#                 self.actor.optimizer.zero_grad()
#                 self.critic.optimizer.zero_grad()
#                 total_loss.backward()
#                 self.actor.optimizer.step()
#                 self.critic.optimizer.step()

#         self.memory.clear_memory()               



#     # class Agent:
#     #     def __init__(self,n_actions,input_dims,gamma=0.99,alpha=0.0003,gae_lambda = 0.95, policy_clip = 0.2, batch_size = 64,N = 2048,n_epochs = 10):
#     #         self.gamma = gamma
#     #         self.policy_clip = policy_clip
#     #         self.n_epochs = n_epochs
#     #         self.gae_lambda = gae_lambda

#     #         self.actor = ActorNetwork(n_actions, input_dims ,alpha)
#     #         self.critic = CriticNetwork(input_dims,alpha)
#     #         self.memory = PPOMemory(batch_size)

#     #     def remember(self, state, action, probs, vals, reward, dones):
#     #         self.memory.store_memory(state,action,probs,vals,reward,dones)
             
#     #     def save_models(self):
#     #         print("saving models . . .")
#     #         self.actor.save_checkpoint()
#     #         self.critic.save_checkpoint()

#     #     def load_models(self):
#     #         print("loading models . . .")
#     #         self.actor.load_checkpoint()
#     #         self.critic.load_checkpoint()

#     #     def choose_action(self, observation):
#     #         state = T.tensor([observation],dtype=T.float).to(self.actor.device)

#     #         dist = self.actor(state)
#     #         value = self.critic(state)
#     #         action = dist.sample()

#     #         probs = T.squeeze(dist.log_prob(action)).item()
#     #         action = T.squeeze(action).item()
#     #         value = T.squeeze(value).item()

#     #         return probs, action, value
        
#     #     def learn(self):
#     #         for _ in range(self.n_epochs):
#     #             state_arr, action_arr, old_probs_arr, vals_arr,\
#     #             reward_arr, done_arr, batches = \
#     #             self.memory.generate_batches()

#     #             values = vals_arr
#     #             advantage = np.zeros(len(reward_arr),dtype = np.float32)

#     #             for t in range(len(reward_arr)-1):
#     #                 discount = 1
#     #                 a_t = 0
#     #                 for k in range(t,len(reward_arr)-1):
#     #                     a_t += discount*(reward_arr[k]+self.gamma*values[k+1]*\
#     #                     (1-int(done_arr[k]))-values[k])
#     #                     discount *= self.gamma*self.gae_lambda
#     #                 advantage[t] = a_t
#     #                 advantage = T.tensor(advantage).to(self.actor.device)
#     #                 values = T.tensor(values).to(self.actor.device)
#     #         for batch in batches:
#     #             states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
#     #             old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
#     #             actions = T.tensor(action_arr[batch]).to(self.actor.device)

#     #             dist = self.actor(states)
#     #             critic_value = self.critic(states)

#     #             critic_value = T.squeeze(critic_value)

#     #             new_probs = dist.log_prob(actions)
#     #             prob_ratio = new_probs.exp() / old_probs.exp()
#     #             #prob_ratio = (new_probs - old_probs).exp()
#     #             weighted_probs = advantage[batch] * prob_ratio
#     #             weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
#     #                     1+self.policy_clip)*advantage[batch]
#     #             actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

#     #             returns = advantage[batch] + values[batch]
#     #             critic_loss = (returns-critic_value)**2
#     #             critic_loss = critic_loss.mean()

#     #             total_loss = actor_loss + 0.5*critic_loss
#     #             self.actor.optimizer.zero_grad()
#     #             self.critic.optimizer.zero_grad()
#     #             total_loss.backward()
#     #             self.actor.optimizer.step()
#     #             self.critic.optimizer.step()

#     #         self.memory.clear_memory()               

                
