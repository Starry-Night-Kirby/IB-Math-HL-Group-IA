from stable_baselines3.common.env_checker import check_env
from graphEnv import GraphEnv


env = GraphEnv()
print("ENV IS BEING CHECKED . . .")
# It will check your custom environment and output additional warnings if needed
check_env(env)