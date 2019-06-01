import copy
import os
import drawSvg as draw
from random import uniform
import numpy as np
from start_project.discriminator.discriminator import Discriminator
from start_project.paint_tool.image_processor import image_to_binary


class PaintEnvParameters(object):
    PREDICTION_REWARD_THRESHOLD = 0.5
    MAX_NO_STEP_PER_EPISODE = 2
    CANVAS_HEIGHT = 100
    CANVAS_WIDTH = 100
    STROKE_WIDTH = 1
    STROKE_COLOR = 'black'
    ACTION_LEN = 2


class PaintEnv(object):

    def __init__(self):
        self.canvas = None
        self.discriminator = Discriminator((28, 28, 1))
        self.current_no_steps = 0
        self.max_no_steps_per_episode = PaintEnvParameters.MAX_NO_STEP_PER_EPISODE

    def reset(self):
        self.canvas = draw.Drawing(PaintEnvParameters.CANVAS_WIDTH, PaintEnvParameters.CANVAS_HEIGHT)
        self.path = draw.Path(stroke_width=PaintEnvParameters.STROKE_WIDTH, stroke=PaintEnvParameters.STROKE_COLOR,
                              fill='none')
        self.path.M(0, 0)

    def step(self, action):
        '''
        action = [intermediate_point, final_point]
        '''
        if len(action) != PaintEnvParameters.ACTION_LEN:
            return
        self.path.Q(*(action[0] + action[1]))
        reward = 0
        self.current_no_steps += 1
        done = self.current_no_steps >= self.max_no_steps_per_episode
        if done:
            self.save_png()
            binary_img = image_to_binary(os.path.dirname(os.path.realpath(__file__)) + '/temp_paint.png')
            prediction = self.discriminator.model.predict(binary_img)[0]
            print(round(prediction[0],4),  round(prediction[1],4))
            if np.argmax(prediction) == 1 and prediction[1] - prediction[0] > PaintEnvParameters.PREDICTION_REWARD_THRESHOLD: #save as constant
                reward = 1 + prediction[1] - prediction[0]
            self.discriminator.train(binary_img)

        return self.path.args['d'], reward, done, {}

    def get_random_action(self):
        return [
            [round(uniform(0, PaintEnvParameters.CANVAS_WIDTH), 2),
             round(uniform(0, PaintEnvParameters.CANVAS_HEIGHT), 2)]
            for _ in range(2)
        ]

    def render(self):
        self.canvas.append(self.path)
        self.canvas.setRenderSize(PaintEnvParameters.CANVAS_WIDTH, PaintEnvParameters.CANVAS_HEIGHT)
        self.canvas.saveSvg('example.svg')

    def save_png(self):
        temp_canvas = copy.deepcopy(self.canvas)
        temp_canvas.append(self.path)
        temp_canvas.setRenderSize(28, 28)
        temp_canvas.savePng('temp_paint.png')
