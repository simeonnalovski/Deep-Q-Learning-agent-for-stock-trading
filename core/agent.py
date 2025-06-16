from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from  keras.optimizers.schedules import InverseTimeDecay
from _collections import deque

import numpy as np
import random


from tensorflow.python.keras.initializers.initializers_v2 import HeUniform, GlorotUniform


class DQNAgent:
    def __init__(self, state_size: int, action_size: int, storage=False):
        self.state_size = state_size  # vlezni nevroni
        self.action_size = action_size  # izlezni nevroni
        self.memory = deque(maxlen=20_000)  # memorija za replay

        self.gamma = 0.94 # idna nagrada
        self.epsilon = 0.74  # stapka na istrazuvanje
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.985
        self.learning_rate = 0.8



        if not storage:
            self.model = self._create_model()
        else:
            self.model = None
        self.from_storage = storage

    def load_model(self, path: str):
        self.model = load_model('models/' + path)
        self.from_storage = True

    def _create_model(self):

        lr_schedule = InverseTimeDecay(
            self.learning_rate,
            decay_rate=0.01,
            decay_steps=1000
        )

        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu', kernel_initializer= HeUniform()))
        model.add(Dense(64, activation='relu',kernel_initializer= HeUniform()))
        model.add(Dense(32, activation='relu',kernel_initializer= HeUniform()))
        model.add(Dense(self.action_size, activation='softmax',kernel_initializer= GlorotUniform()))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=lr_schedule))
        return model

    def remember(self, state: [[]], action: int, reward: float, next_state: [[]],
                 done: bool):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: [[]]):
        # epsilon greedy, za evaluacija, ne za istrazuvanje
        if not self.from_storage and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size: int):
        if len(self.memory) < batch_size:
            return

        # expirience replay
        for i in range(1, 4):
            minibatch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    # Q(s',a)
                    target = (reward + self.gamma *
                              np.amax(self.model.predict(next_state)[0]))
                # Q(s,a)
                target_f = self.model.predict(state)
                # aproksimalno mapiranje na dejstvo
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)
        # opaganje na epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
