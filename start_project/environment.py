import random, numpy, math

from start_project.paint_tool.paint_env import PaintEnv


class Environment:
    def __init__(self):
        self.env = PaintEnv()
        self.env._max_episode_steps = 8000


    def run(self, agent):
        s = self.env.reset()
        R = 0

        while True:
            # self.env.render()

            a = agent.act(s)

            s_, r, done, info = self.env.step(a)

            if done:  # terminal state
                s_ = None

            agent.observe((s, a, r, s_))
            agent.replay()

            s = s_
            R += r

            if done:
                break

        print("Total reward:", R)
