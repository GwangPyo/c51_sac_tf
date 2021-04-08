
def evaluate(env, model, steps=10000):
    scores = []
    for _ in range(steps):
        obs = env.reset()
        episode_score = 0
        while True:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_score += reward
            if done:
                break

        scores.append(episode_score)
        print("episode_score", episode_score)
    return scores


if __name__ == "__main__":
    from c51_sac.c51sac import C51SAC
    from stable_baselines import SAC
    import gym
    import numpy as np
    env = gym.make("LunarLanderContinuous-v2")
    model = C51SAC.load("c51_sac.zip")
    scores = evaluate(env, model)
    print(np.mean(scores))