import gym
import numpy as np
from ppo_torch import Agent
from utils import plot_learning_curve
import matplotlib.pyplot
from graphEnv import GraphEnv

if __name__ == '__main__':
    env = GraphEnv(shape_type="rect")
    env.reset()
    N = 20
    batch_size = 6
    n_epochs = 4
    alpha =0.0001
    agent = Agent(n_actions=env.action_space.shape[0],batch_size=batch_size,alpha = alpha, n_epochs = n_epochs, input_dims = env.observation_space.shape)
    n_games = 200
    figure_file = 'shape_packing.png'
    best_score = env.reward
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    
    for i in range(n_games):
        observation,_ = env.reset()
        done = False
        score = 0
        while not done:
            # print("[DEBUG] Choosing action...")
            action, prob, val = agent.choose_action(observation)
            # print("[DEBUG] Action chosen:", action)
            # print(i)
            # print("[DEBUG MAIN] action to env.steps: ",action,", shape: ",getattr(action,"shape","N/A"))
            observation, reward, done,truncated, info = env.step(action)
            done = done or truncated
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            
            n_steps += 1
            
            if n_steps % N ==0:
                agent.learn()
                learn_iters += 1
            observation = env._get_obs()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
        if score > best_score:
           best_score = score
           agent.save_models()

        
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
    env.render()
    env.save_best_attempt("final_result.png")

            

