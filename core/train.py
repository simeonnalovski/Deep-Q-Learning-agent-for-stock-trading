from core.util import create_label, visualize_profit, visualize_rewards
from core.agent import *
from core.env import create_environment
import os

# train se koristi za treniranje i zacuvucanje na profit
def train(data: str):
    env = create_environment(data)
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    epochs = 100
    rewards = []
    profits = []
    print('epochs: ' + str(epochs))
    print('start training:')
    for e in range(epochs):
        info = play(agent, env)
        # zacuvuvanje na vkupnite profiti i nagradi
        rewards.append(env.get_total_reward())
        profits.append(env.get_total_profit())
        # pecatenje na krajnite profit i nagrada po epoha
        print(info)
    ticker= os.path.splitext(data)[0].split('_')[0]
    # zacuvuvanje na modelot
    agent.model.save("models/model_" +ticker+ create_label()+'.keras')
    calc_results_and_update(profits, rewards,ticker)


def play(agent, env):
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    # reset na sostojba prinova epizoda
    state = env.reset()
    # flatten
    state = np.reshape(state, [1, state_size])
    done = False
    info = None
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        # zapamtuvanje na prethodnite parametri
        agent.remember(state, action, reward, next_state, done)

        # kraj na epoch pri pad na vkupniot profit
        if env.get_total_profit() < 0.8:
            break
        # novata sostojba stanuva pocetna sostojba na nova epizoda
        state = next_state
    # povtorno treniranje so replay
    agent.replay(64)
    return info
# presmetka na prosecna nagrada i profit
def calc_results_and_update(profits: [float], rewards: [float],ticker:str):

    mean_profit = np.mean(profits)
    print("Average profit:", mean_profit)
    visualize_profit(profits,ticker)

    mean_reward = np.mean(rewards)
    print("Average reward:", mean_reward)
    visualize_rewards(rewards,ticker)
