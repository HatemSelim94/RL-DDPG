import torch
from collections import deque
import numpy as np


def ddpg_train(agent, env, brain_name, n_episodes=2000, max_t=300000, print_every=100):
    """
      :agent (DDPG agent): DDPG agent
      :env (Unity environment): Unity environment(Reacher 1 agent)
      :n_episodes (int): Number of training episodes
      :max_t (int): Max. steps per episode
      :print_every(int): Frequncy of printing the avg. score
    """
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes+1):
        #state = env.reset()     # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]                  # get the current state
        agent.reset()
        score = 0
        for t in range(max_t):
            actions = agent.act(state, i_episode)
            env_info = env.step(actions)[brain_name] 
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, actions, reward, next_state, done) # makes the agent learn
            state = next_state
            score += reward
            if done:
                break
        scores_deque.append(score)
        scores.append(score) 
        print('\rEpisode {}\t Episode score: {:.2f}'.format(i_episode,score), end="")
        if i_episode % print_every == 0:
            avg_score = np.mean(scores_deque)
            print('\rEpisode {}\tAverage Score: {:.2f}\t Episode score: {:.2f}'.format(i_episode, avg_score, score))
            if avg_score >= 30:
                actor_training_state = {'episode': i_episode, 
                              'agent_actor_dict': agent.actor_local.state_dict(),
                              'agent_actor_optimizer':agent.actor_optimizer.state_dict()}
                critic_training_state = {'episode': i_episode, 
                              'agent_critic_dict': agent.critic_local.state_dict(),
                              'agent_critic_optimizer':agent.critic_optimizer.state_dict()}
                torch.save(actor_training_state, 'checkpoint_actor.pth')
                torch.save(critic_training_state, 'checkpoint_critic.pth')
                break
    return scores