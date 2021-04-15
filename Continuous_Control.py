from unityagents import UnityEnvironment
from ddpg import Agent
from ddpg_train import ddpg_train
import matplotlib.pyplot as plt

# Unity environment
env = UnityEnvironment(file_name='/data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
states = env_info.vector_observations
state_size = states.shape[1]
action_size = brain.vector_action_space_size

#DDPG Agent
agent = Agent(state_size=state_size, action_size=action_size)
scores = ddpg_train(agent, env, brain_name)

# plot
episodes = list(range(1, len(scores)+1))
plt.plot(episodes, scores,marker='*')
plt.title("DDPG-Reacher")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.axvline(episodes[-1]-100,color='g')
plt.axhline(30,color='g')
plt.savefig('DDPG_Reacher_score.png', bbox_inches='tight')