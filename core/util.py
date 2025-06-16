import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym_anytrading.envs import TradingEnv
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler



def create_label():
    date = datetime.now()
    return date.strftime("%y_%m_%d")


def load_data(path: str):
    data = None
    try:
        data = pd.read_csv('data/' + path)
    except FileNotFoundError:
        print('File for that stock does not exist in the system')
    return data


def process_data(df: pd.DataFrame, window_size: int, frame_bound: tuple):
    assert df.ndim == 2
    assert df.shape[1] == 7
    assert 'Close' in df
    start = frame_bound[0] - window_size
    end = frame_bound[1]
    prices = df.loc[:, 'Close'].to_numpy()[start:end]
    signal_features = df.loc[:, ['Close']].to_numpy()[start:end]
    return prices, signal_features


def standard_scale(window: [[int]]):
    window = StandardScaler().fit_transform(window)
    return window


def min_max_scale(window: [[int]]):
    window = MinMaxScaler(feature_range=(0.1, 1)).fit_transform(window)
    return window


def sigmoid(x: int):
    return 1 / (1 + np.exp(-x))


def sigmoid_scale(window: [[int]]):
    window = [i for j in window for i in j]
    window = [sigmoid(i) for i in window]
    return np.reshape([[i] for i in window], (-1, 1))


#vcituvanje i azuriranje na prerformans
def update_performance(mean_profit: float, mean_reward: float, last_profit: float, last_reward: float):
    performance = pd.read_csv("logs/performance.csv")
    performance.loc[len(performance)] = [len(performance), create_label(), mean_profit, mean_reward, last_profit,
                                         last_reward]
    performance.to_csv("logs/performance.csv", index_label=False)

def visualize_data(y, ylabel, title, label,tick:str):
    x_values = range(0, len(y), 1)
    y_values = y
    plt.plot(x_values, y_values)
    plt.xlabel('episodes')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig("plots/"+label+"/" +label+tick+ create_label() + ".png")
    plt.show()

def visualize_profit(total_profit: [float],tick:str):
    visualize_data(total_profit, 'cumulative profit', 'Profit by Episode',
                   'profit',tick)
def visualize_rewards(total_rewards: [float],tick:str):
    visualize_data(total_rewards, 'cumulative rewards', 'Reward by Episode',
                   'reward',tick)
def visualize_trades(env: TradingEnv, save: bool, model: str):
    plt.cla()
    env.render_all()
    if save:
        plt.savefig("plots/trades/trades_" + model + ".png")
    plt.show()
