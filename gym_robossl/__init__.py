import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='SSLNav-v0',
    entriy_point='gym_robossl.envs:',
    timestep_limit='5000',
    reward_threshold=10.0,
    nondeterministic=True,
)
