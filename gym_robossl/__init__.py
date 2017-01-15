import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='SSLNav-v0',
    entry_point='gym_robossl.envs:SSLSimpleNav',
    timestep_limit='5000',
    reward_threshold=10.0,
    nondeterministic=True,
)

register(
    id='SSLNavCont-v0',
    entry_point='gym_robossl.envs:SSLSimpleNavContinuos',
    timestep_limit='5000',
    reward_threshold=10.0,
    nondeterministic=True,
)
