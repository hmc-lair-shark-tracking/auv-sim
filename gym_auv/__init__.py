from gym.envs.registration import register

register(id = 'auv-v0', entry_point = 'gym_auv.envs:AuvEnv')
