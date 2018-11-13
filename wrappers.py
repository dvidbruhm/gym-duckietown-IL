import gym
from gym import spaces
import numpy as np

def steering_to_wheel(action):
    gain=1.0
    trim=0.0
    radius=0.0318
    k=27.0
    limit=1.0
    wheel_dist=0.102
    vel, angle = action

    # assuming same motor constants k for both motors
    k_r = k
    k_l = k

    # adjusting k by gain and trim
    k_r_inv = (gain + trim) / k_r
    k_l_inv = (gain - trim) / k_l

    omega_r = (vel + 0.5 * angle * wheel_dist) / radius
    omega_l = (vel - 0.5 * angle * wheel_dist) / radius

    # conversion from motor rotation rate to duty cycle
    u_r = omega_r * k_r_inv
    u_l = omega_l * k_l_inv

    # limiting output to limit, which is 1.0 for the duckiebot
    u_r_limited = max(min(u_r, limit), -limit)
    u_l_limited = max(min(u_l, limit), -limit)

    vels = np.array([u_l_limited, u_r_limited])
    return vels
