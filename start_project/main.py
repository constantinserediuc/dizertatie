from start_project.agent import Agent
from start_project.environment import Environment
from start_project.paint_tool.paint_env import PaintEnv

env = PaintEnv()
def do_epoch():
    env.reset()
    for _ in range(4):
        s, r, done, info = env.step(env.get_random_action())  # take a random action
        # print(s)
        # print(r)
        # print(done)
    env.render()
# for _ in range(100):
#     do_epoch()


env = Environment()

stateCnt = 1#env.env.observation_space.shape[0] #string
actionCnt = env.env.action_space.n

agent = Agent(stateCnt, actionCnt)

try:
    t = 0
    while True:
        print(t)
        t +=1
        env.run(agent)
finally:
    agent.brain.model.save("dqn.h5")

