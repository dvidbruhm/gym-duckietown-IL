import cv2
from gym_duckietown.envs import DuckietownEnv
from teacher import PurePursuitExpert
from _loggers import Logger

import pygame


USER_INPUT = False
key = "no input"

if USER_INPUT:
    pygame.init()
    windowSurface = pygame.display.set_mode((50, 50), 0, 32)
    windowSurface.fill((0,0,0))


def get_user_direction():
    global key
    
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                key = "left"
            elif event.key == pygame.K_d:
                key = "right"
        if event.type == pygame.KEYUP:
            key = "no input"

    return key



# Log configuration, you can pick your own values here
# the more the better? or the smarter the better?
EPISODES = 200
STEPS = 300

DEBUG = False

env = DuckietownEnv(
    map_name='udem1',  # check the Duckietown Gym documentation, there are many maps of different complexity
    max_steps=EPISODES * STEPS
)

# this is an imperfect demonstrator... I'm sure you can construct a better one.
expert = PurePursuitExpert(env=env)

# please notice
logger = Logger(env, log_file='train.log')

# let's collect our samples
for episode in range(0, EPISODES):
    for steps in range(0, STEPS):
        
        if USER_INPUT:
            key = get_user_direction()
        
        try:
            # we use our 'expert' to predict the next action.
            action = expert.predict(None, user_input=key)
            observation, reward, done, info = env.step(action)
            # we can resize the image here
            observation = cv2.resize(observation, (80, 60))
            # NOTICE: OpenCV changes the order of the channels !!!
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
        except:
            print("ERRRRRRRRRRRRRRRRRROR")
            break
        # we may use this to debug our expert.
        if DEBUG:
            cv2.imshow('debug', observation)
            cv2.waitKey(1)

        logger.log(observation, action, reward, done, info)
        # [optional] env.render() to watch the expert interaction with the environment
        # we log here
    logger.on_episode_done()  # speed up logging by flushing the file
    env.reset()

# we flush everything and close the file, it should be ~ 120mb
# NOTICE: we make the log file read-only, this prevent us from erasing all collected data by mistake
# believe me, this is an important issue... can you imagine loosing 2 GB of data? No? We do...
logger.close()

env.close()
