import numpy as np
from tqdm import tqdm
from agent import Agent
from game import Action


class CardGame:
    def __init__(self,
                 agent1: Agent = None,
                 agent2: Agent = None,):
        self.agent1 = agent1
        self.agent2 = agent2
        self.bb_total = np.zeros((13, 51))
        self.sb_total = np.zeros((13, 51))
        self.bb_push_count = np.zeros((13, 51))
        self.sb_push_count = np.zeros((13, 51))

    def play_episode(self):
        effective_stack = np.random.randint(10, 51)
        card1 = np.random.randint(0, 13)
        card2 = np.random.randint(0, 13)

        big_blind = self.agent1 if np.random.rand() < 0.5 else self.agent2
        small_blind = self.agent1 if big_blind == self.agent2 else self.agent2
        action = small_blind.choose_action(card1, effective_stack, True) 
        self.sb_total[card1, effective_stack] += 1

        if action == Action.PUSH:
            action = big_blind.choose_action(card2, effective_stack, False)
            self.bb_total[card2, effective_stack] += 1
            self.sb_push_count[card1, effective_stack] += 1

            if action == Action.PUSH:
                self.bb_push_count[card2, effective_stack] += 1
                if card1 > card2:
                    small_blind.feed_reward(card1,
                                            effective_stack,
                                            Action.PUSH,
                                            2 / effective_stack,
                                            True)
                    big_blind.feed_reward(card2,
                                          effective_stack,
                                          Action.PUSH,
                                          -2 / effective_stack,
                                          False)
                else:
                    small_blind.feed_reward(card1,
                                            effective_stack,
                                            Action.PUSH,
                                            -1 / effective_stack,
                                            True)
                    big_blind.feed_reward(card2,
                                          effective_stack,
                                          Action.PUSH,
                                          1 / effective_stack,
                                          False)
            else:
                small_blind.feed_reward(card1,
                                        effective_stack,
                                        Action.PUSH,
                                        2 / effective_stack,
                                        True)
                big_blind.feed_reward(card2,
                                      effective_stack,
                                      Action.FOLD,
                                      -2 / effective_stack,
                                      False)
        else:
            small_blind.feed_reward(card1,
                                    effective_stack,
                                    Action.FOLD,
                                    -1 / effective_stack,
                                    True)

    def get_play_stats(self):
        bb_stats = (self.bb_push_count / self.bb_total) * 100.0
        sb_stats = (self.sb_push_count / self.sb_total) * 100.0
        return bb_stats, sb_stats

    def play(self, n=1000):
        for _ in tqdm(range(n)):
            self.play_episode()
