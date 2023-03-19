import logging
import coloredlogs
import pickle

import aicrowd_gym
import minerl

import cv2
import os
import numpy as np

from config import EVAL_EPISODES, EVAL_MAX_STEPS
from openai_vpt.agent import MineRLAgent

EVAL_MAX_STEPS = 18000 #Overwrite max steps to play for 15 minutes
EVAL_EPISODES = 1 #overwrite episodes
coloredlogs.install(logging.DEBUG)

MINERL_GYM_ENV = 'MineRLObtainDiamondShovel-v0'
MODEL = 'data/VPT-models/2x.model'
WEIGHTS = 'data/VPT-models/rl-from-early-game-2x.weights'


def main():
    # NOTE: It is important that you use "aicrowd_gym" instead of regular "gym"!
    #       Otherwise, your submission will fail.
    env = aicrowd_gym.make(MINERL_GYM_ENV)

    # Load the model
    agent_parameters = pickle.load(open(MODEL, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(WEIGHTS)

    for i in range(EVAL_EPISODES):
        obs = env.reset()
        agent.reset()
        frameSize = (640,360)
        output = cv2.VideoWriter(r"P:\Programming\MineRL\basalt-2022-intro-track-baseline\frames\output.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 20, frameSize)
        for step_counter in range(EVAL_MAX_STEPS):

            # Step your model here.
            minerl_action = agent.get_action(obs)

            obs, reward, done, info = env.step(minerl_action)

            # Uncomment the line below to see the agent in action:
            image=env.render()
            #print(type(image))
            #print(image.shape)
            #image = obs["pov"]
            image = image[..., ::-1] #fix RGB
            output.write(image)

            if done:
                output.release()
                break
        print(f"[{i}] Episode complete")

    # Close environment and clean up any bigger memory hogs.
    # Otherwise, you might start running into memory issues
    # on the evaluation server.
    env.close()


if __name__ == "__main__":
    main()
