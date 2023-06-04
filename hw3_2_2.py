# A variation of Zeynep Cankara's GitHub repository
import numpy as np
import matplotlib.pyplot as plt


# Creates a table of Q_values (state-action) initialized with zeros
# Initialize Q(s, a), for all s ∈ S, a ∈ A(s), arbitrarily, and Q(terminal-state, ·) = 0.
def createQ_table(rows=4, cols=12):
    # initialize the q_table with all zeros for each state and action
    q_table = np.zeros((4, cols * rows))
    return q_table


# Choosing action using policy
# Sutton's code pseudocode: Choose A from S using policy derived from Q (e.g., ε-greedy)
# 10% exploration to avoid getting stuck at a local optima
def epsilon_greedy_policy(state, q_table, epsilon=0.1):
    # choose a random int from a uniform distribution [0.0, 1.0)
    decide_explore_exploit = np.random.random()

    if decide_explore_exploit < epsilon:
        # UP = 0, LEFT = 1, RIGHT = 2, DOWN = 3
        action = np.random.choice(4)
    else:
        # Choose the action with largest Q-value (state value)
        action = np.argmax(q_table[:, state])

    return action


def move_agent(agent, action):
    # get position of the agent
    (posX, posY) = agent
    # UP
    if (action == 0) and posX > 0:
        posX = posX - 1
    # LEFT
    if (action == 1) and (posY > 0):
        posY = posY - 1
    # RIGHT
    if (action == 2) and (posY < 11):
        posY = posY + 1
    # DOWN
    if (action == 3) and (posX < 3):
        posX = posX + 1
    agent = (posX, posY)

    return agent


def get_state(agent, q_table):
    # get position of the agent
    (posX, posY) = agent

    # obtain the state value
    state = 12 * posX + posY

    # get maximum state value from the table
    state_action = q_table[:, int(state)]
    # return the state value with for the highest action
    maximum_state_value = np.amax(state_action)
    return state, maximum_state_value


def get_reward(state):
    # game continues
    game_end = False
    # all states except cliff have -1 value
    reward = -1
    # goal state
    if state == 47:
        game_end = True
        reward = 10
    # cliff
    if 37 <= state <= 46:
        game_end = True
        # Penalize the agent if agent encounters a cliff
        reward = -100

    return reward, game_end


def update_qTable(q_table, state, action, reward, next_state_value, gamma_discount=0.9, alpha=0.5):
    update_q_value = q_table[action, state] + alpha * (reward + (gamma_discount * next_state_value) - q_table[
        action, state])
    q_table[action, state] = update_q_value

    return q_table


def qlearning(num_episodes=500, gamma_discount=0.9, alpha=0.5, epsilon=0.1):
    # initialize all states to 0
    # Terminal state cliff_walking ends
    reward_cache = list()
    step_cache = list()
    q_table = createQ_table()
    # starting from left down corner
    agent = (3, 0)
    # start iterating through the episodes
    for episode in range(0, num_episodes):
        env = np.zeros((4, 12))
        env = visited_env(agent, env)
        # starting from left down corner
        agent = (3, 0)
        game_end = False
        # cumulative reward of the episode
        reward_cum = 0
        # keeps number of iterations until the end of the game
        step_cum = 0
        while not game_end:
            # get the state from agent's position
            state, _ = get_state(agent, q_table)
            # choose action using epsilon-greedy policy
            action = epsilon_greedy_policy(state, q_table)
            # move agent to the next state
            agent = move_agent(agent, action)
            step_cum += 1
            env = visited_env(agent, env)  # mark the visited path
            # observe next state value
            next_state, max_next_state_value = get_state(agent, q_table)
            # observe reward and determine whether game ends
            reward, game_end = get_reward(next_state)
            reward_cum += reward
            # update q_table
            q_table = update_qTable(q_table, state, action, reward, max_next_state_value, gamma_discount, alpha)
        reward_cache.append(reward_cum)
        if episode > 498:
            print("Agent trained with Q-learning after 500 iterations")
            # display the last 2 path agent takes
            print(env)
        step_cache.append(step_cum)
    return q_table, reward_cache, step_cache


def sarsa(num_episodes=500, gamma_discount=0.9, alpha=0.5):
    # initialize all states to 0
    # Terminal state cliff_walking ends
    q_table = createQ_table()
    step_cache = list()
    reward_cache = list()
    # start iterating through the episodes
    for episode in range(0, num_episodes):
        # starting from left down corner
        agent = (3, 0)
        game_end = False
        # cumulative reward of the episode
        reward_cum = 0
        # keeps number of iterations until the end of the game
        step_cum = 0
        env = np.zeros((4, 12))
        env = visited_env(agent, env)
        # choose action using policy
        state, _ = get_state(agent, q_table)
        action = epsilon_greedy_policy(state, q_table)
        while not game_end:
            # move agent to the next state
            agent = move_agent(agent, action)
            env = visited_env(agent, env)
            step_cum += 1
            # observe next state value
            next_state, _ = get_state(agent, q_table)
            # observe reward and determine whether game ends
            reward, game_end = get_reward(next_state)
            reward_cum += reward
            # choose next_action using policy and next state
            next_action = epsilon_greedy_policy(next_state, q_table)
            # update q_table. differs from q-learning uses the next action determined by policy
            next_state_value = q_table[next_action][next_state]
            q_table = update_qTable(q_table, state, action, reward, next_state_value, gamma_discount, alpha)
            # update the state and action
            state = next_state
            # differs q_learning both state and action must update
            action = next_action
        reward_cache.append(reward_cum)
        step_cache.append(step_cum)
        if episode > 498:
            print("Agent trained with SARSA after 500 iterations")
            # display the last 2 path agent takes
            print(env)
    return q_table, reward_cache, step_cache


def visited_env(agent, env):
    # Visualize the path agent takes
    (posY, posX) = agent
    env[posY][posX] = 1
    return env


def retrieve_environment(q_table, action):
    # Displays the environment state values for a specific action. Implemented for debug purposes
    # Args: q_table -- type(np.array) Determines state value.
    # action -- type(int) action value [0:3] -> [UP, LEFT, RIGHT, DOWN]
    env = q_table[action, :].reshape((4, 12))
    # display environment values
    print(env)


def plot_cumreward_normalized(reward_cache_qlearning, reward_cache_SARSA):
    # Visualizes the reward convergence. Args: reward_cache -- type(list) contains cumulative_reward
    cum_rewards_q = []
    rewards_mean = np.array(reward_cache_qlearning).mean()
    rewards_std = np.array(reward_cache_qlearning).std()
    # used to determine the batches
    count = 0
    # accumulate reward for the batch
    cur_reward = 0
    for cache in reward_cache_qlearning:
        count = count + 1
        cur_reward += cache
        if count == 10:
            # normalize the sample
            normalized_reward = (cur_reward - rewards_mean) / rewards_std
            cum_rewards_q.append(normalized_reward)
            # cum_rewards_q.append(cur_reward)
            cur_reward = 0
            count = 0

    cum_rewards_SARSA = []
    rewards_mean = np.array(reward_cache_SARSA).mean()
    rewards_std = np.array(reward_cache_SARSA).std()
    # used to determine the batches
    count = 0
    # accumulate reward for the batch
    cur_reward = 0
    for cache in reward_cache_SARSA:
        count = count + 1
        cur_reward += cache
        if count == 10:
            # normalize the sample
            normalized_reward = (cur_reward - rewards_mean) / rewards_std
            cum_rewards_SARSA.append(normalized_reward)
            # cum_rewards_SARSA.append(cur_reward)
            cur_reward = 0
            count = 0

    # prepare the graph
    plt.plot(cum_rewards_q, label="q_learning")
    plt.plot(cum_rewards_SARSA, label="SARSA")
    plt.ylabel('Cumulative Rewards')
    plt.xlabel('Batches of Episodes (sample size 10) ')
    plt.title("Q-Learning/SARSA Convergence of Cumulative Reward")
    plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
    plt.show()


def plot_number_steps(step_cache_qlearning, step_cache_SARSA):
    # Visualize number of steps taken
    cum_step_q = []
    steps_mean = np.array(step_cache_qlearning).mean()
    steps_std = np.array(step_cache_qlearning).std()
    # used to determine the batches
    count = 0
    # accumulate reward for the batch
    cur_step = 0
    for cache in step_cache_qlearning:
        count = count + 1
        cur_step += cache
        if count == 10:
            # normalize the sample
            normalized_step = (cur_step - steps_mean) / steps_std
            cum_step_q.append(normalized_step)
            # cum_step_q.append(cur_step)
            cur_step = 0
            count = 0

    cum_step_SARSA = []
    steps_mean = np.array(step_cache_SARSA).mean()
    steps_std = np.array(step_cache_SARSA).std()
    # used to determine the batches
    count = 0
    # accumulate reward for the batch
    cur_step = 0
    for cache in step_cache_SARSA:
        count = count + 1
        cur_step += cache
        if count == 10:
            # normalize the sample
            normalized_step = (cur_step - steps_mean) / steps_std
            cum_step_SARSA.append(normalized_step)
            # cum_step_SARSA.append(cur_step)
            cur_step = 0
            count = 0

    # prepare the graph
    plt.plot(cum_step_q, label="q_learning")
    plt.plot(cum_step_SARSA, label="SARSA")
    plt.ylabel('Number of iterations')
    plt.xlabel('Batches of Episodes (sample size 10) ')
    plt.title("Q-Learning/SARSA Iteration number until game ends")
    plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
    plt.show()


def plot_qlearning_smooth(reward_cache):
    mean_rev = (np.array(reward_cache[0:11]).sum()) / 10
    # initialize with cache mean
    cum_rewards = [mean_rev] * 10
    idx = 0
    for cache in reward_cache:
        cum_rewards[idx] = cache
        idx += 1
        smooth_reward = (np.array(cum_rewards).mean())
        cum_rewards.append(smooth_reward)
        if idx == 10:
            idx = 0

    plt.plot(cum_rewards)
    plt.ylabel('Cumulative Rewards')
    plt.xlabel('Batches of Episodes (sample size 10) ')
    plt.title("Q-Learning  Convergence of Cumulative Reward")
    plt.legend(loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
    plt.show()


def main():
    # Learn state dynamics obtain cumulative rewards for 500 episodes
    # SARSA
    q_table_SARSA, reward_cache_SARSA, step_cache_SARSA = sarsa()
    # QLEARNING
    q_table_qlearning, reward_cache_qlearning, step_cache_qlearning = qlearning()

    plot_number_steps(step_cache_qlearning, step_cache_SARSA)
    # Visualize the result
    plot_cumreward_normalized(reward_cache_qlearning, reward_cache_SARSA)


if __name__ == "__main__":
    # call main function to execute grid world
    main()
