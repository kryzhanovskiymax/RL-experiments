import collections

Experience = collections.namedtuple(
    'Experience',
    field_names=['state', 'action', 'reward', 'next_state', 'done']
)

class ExperienceBuffer():
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(buffer)

    def append(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = \
                zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), \
               np.array(actions), \
               np.array(rewards), \
               np.array(next_states), \
               np.array(dones)