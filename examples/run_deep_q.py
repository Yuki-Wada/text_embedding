import gym
env = gym.make('CartPole-v0')
for i in range(10):
    observation = env.reset()
    for t in range(1000):
        env.render()
        # observation:
        #       カート位置 -2.4 - 2.4
        #       カート速度 -3.0 - 3.0
        #       棒の角度   -41.8度 - 41.8度
        #       棒の角速度 -2.0 - 2.0
        # 終了条件
        #       ポールが垂直から15度 (=15 / 2 / pi) を超える
        #       カートが中心から2.4ユニット以上移動する
        observation, reward, done, info = env.step(env.action_space.sample())
        print(observation, reward, done, info)
        if done:
            print("Episode{} finished after {} timesteps".format(i, t+1))
            break
env.close()
