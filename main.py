from sample_env import freq_env
from DQN import DeepQNetwork

def run_dqn():
    step = 0
    for episode in range(300):
        # initial observation
        observation = env.reset()
        #print(observation)
        while True:
            # fresh env
            #env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of anti_jamming
    print('anti_jamming over')
    


if __name__ == "__main__":
    # maze game
    env = freq_env()
    env.main_thread()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    run_dqn()
    RL.plot_cost()