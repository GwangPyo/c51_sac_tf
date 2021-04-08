
def evaluate(env, model, steps=10000):
    scores = []
    for _ in range(steps):
        obs = env.reset()
        episode_score = 0
        while True:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step()
            episode_score += reward
            if done:
                break
        scores.append(episode_score)
    return scores