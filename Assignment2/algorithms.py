import numpy as np
import json
import wandb
from collections import deque

from gridworld import GridWorld

# =========================== 2.1 model free prediction ===========================
class ModelFreePrediction:
    """
    Base class for ModelFreePrediction algorithms
    """
       

    def __init__(self, grid_world: GridWorld, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
            seed (int): seed for sampling action from the policy
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.max_episode = max_episode
        self.episode_counter = 0  
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.values       = np.zeros(self.state_space)
        self.rng = np.random.default_rng(seed)      # only call this in collect_data()
        if policy:
            self.policy = policy
        else:
            self.policy = np.ones((self.state_space, self.action_space)) / self.action_space  # random policy

    def get_all_state_values(self) -> np.array:
        return self.values

    def collect_data(self) -> tuple:
        """
        Use the stochastic policy to interact with the environment and collect one step of data.
        Samples an action based on the action probability distribution for the current state.
        """

        current_state = self.grid_world.get_current_state()  # Get the current state
        
        # Sample an action based on the stochastic policy's probabilities for the current state
        action_probs = self.policy[current_state]  
        action = self.rng.choice(self.action_space, p=action_probs)  

        next_state, reward, done = self.grid_world.step(action)  
        if done:
            self.episode_counter +=1
        return next_state, reward, done
        

class MonteCarloPrediction(ModelFreePrediction):
    def __init__(self, grid_world: GridWorld, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """
        Constructor for MonteCarloPrediction
        
        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)

    def run(self) -> None:
        """Run the algorithm until max_episode"""
        # TODO: Update self.values with first-visit Monte-Carlo method
        current_state = self.grid_world.reset()
        returns = np.zeros(self.state_space)
        N_count = np.zeros(self.state_space)
        while self.episode_counter < self.max_episode:
            # generate episode
            episode = []
            done = False
            while not done:
                next_state, reward, done = self.collect_data()
                episode.append((current_state, reward))
                current_state = next_state

            states_in_episode = [x[0] for x in episode]
    
            # visited_states = set()
            G = 0
            for t in range(len(episode)-1, -1, -1): # 反向遍歷，直到算到那個state是第一次在所有state中出現，最後會走回到起點
                state, reward = episode[t]
                G = self.discount_factor*G + reward   
                
                # 這邊的第一次是從頭數過來的第一次，不是反向回來的第一次!
                if t == states_in_episode.index(state):
                    returns[state] += G
                    N_count[state] += 1
                    self.values[state] = returns[state]/N_count[state]                




class TDPrediction(ModelFreePrediction):
    def __init__(
            self, grid_world: GridWorld,learning_rate: float, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
            learning_rate (float): learning rate for updating state value
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)
        self.lr     = learning_rate

    def run(self) -> None:
        """Run the algorithm until max episode"""
        # TODO: Update self.values with TD(0) Algorithm
        current_state = self.grid_world.reset()
        while self.episode_counter < self.max_episode:
            episode = []
            done = False
            while not done:
                next_state, reward, done = self.collect_data()
                episode.append((current_state, reward))
                # 注意done的next state就直接設0了
                self.values[current_state] += self.lr*(reward + self.discount_factor*self.values[next_state]*(1-done) \
                                                       - self.values[current_state])
                current_state = next_state
            # continue


class NstepTDPrediction(ModelFreePrediction):
    def __init__(
            self, grid_world: GridWorld, learning_rate: float, num_step: int, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
            learning_rate (float): learning rate for updating state value
            num_step (int): n_step look ahead for TD
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)
        self.lr     = learning_rate
        self.n      = num_step

    def run(self) -> None:
        """Run the algorithm until max_episode"""
        # TODO: Update self.values with N-step TD Algorithm
        current_state = self.grid_world.reset()
        while self.episode_counter < self.max_episode:
            episode = []
            done = False
            t = 0
            T = float("inf")

            while not done:
                if t < T:
                    next_state, reward, done = self.collect_data()
                    episode.append((current_state, reward, next_state))
                    
                    if done:
                        T = t + 1 # 縮短T的範圍到episode的長度
                tao = t - self.n + 1  # 退回去 n 個 step 前來更新。因為更新值即為該state往前後算n個的reward

                # 代表此時已經走了 n 步以上了，可以開始累加reward
                if tao >= 0:
                    G = 0
                    for i in range(tao, min(tao + self.n, T)):  # 累加 n 個reward，如果到終點了，加到終點就好
                        # print(i)
                        G += (self.discount_factor ** (i - tao)) * episode[i][1]
                    if tao + self.n < T: # 還沒走到終點，把next state value更新進去，走到終點就不用加上next state value
                        G += (self.discount_factor ** self.n) * self.values[episode[t][2]]
                        state_tao = episode[tao][0]
                        self.values[state_tao] += self.lr * (G - self.values[state_tao])

                current_state = next_state
                t += 1

            # 還有剩下的value 沒有被update到，直接單純利用R來更新，跟Monte Carlo一樣
            for tao in range(t - self.n, t):
                G = 0
                for i in range(tao, t):
                    G += self.discount_factor ** (i - tao) * episode[i][1]
                self.values[episode[tao][0]] += self.lr * (G - self.values[episode[tao][0]])


# =========================== 2.2 model free control ===========================
class ModelFreeControl:
    """
    Base class for model free control algorithms 
    """

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.q_values     = np.zeros((self.state_space, self.action_space))  
        self.policy       = np.ones((self.state_space, self.action_space)) / self.action_space # stocastic policy
        self.policy_index = np.zeros(self.state_space, dtype=int)                              # deterministic policy

    def get_policy_index(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy_index
        """
        for s_i in range(self.state_space):
            self.policy_index[s_i] = self.q_values[s_i].argmax()
        return self.policy_index
    
    def get_max_state_values(self) -> np.ndarray:
        max_values = np.zeros(self.state_space)
        for i in range(self.state_space):
            max_values[i] = self.q_values[i].max()
        return max_values



class MonteCarloPolicyIteration(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for MonteCarloPolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon
        # add
        self.learning_curve = []
        self.loss_curve = []

    def policy_evaluation(self, state_trace, action_trace, reward_trace) -> None:
        """Evaluate the policy and update the values after one episode"""
        # TODO: Evaluate state value for each Q(s,a)
        G = 0
        loss_record = []
        for i in range(len(reward_trace)-1, -1, -1):
            reward = reward_trace[i]
            state = state_trace[i]
            action = action_trace[i]
            G = self.discount_factor * G + reward
            loss = G - self.q_values[state][action]
            self.q_values[state][action] += self.lr*loss     
            # add
            loss_record.append(abs(loss))
        return np.mean(loss_record)

        
        # state value最後會get出來，所以不用自己從q value寫到state value

    def policy_improvement(self) -> None:
        """Improve policy based on Q(s,a) after one episode"""
        # TODO: Improve the policy
        for s in range(self.state_space):
            for a in range(self.action_space):
                if a == np.argmax(self.q_values[s]):
                    self.policy[s][a] = self.epsilon/self.action_space + 1 - self.epsilon
                else:
                    self.policy[s][a] = self.epsilon/self.action_space

        # raise NotImplementedError


    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Monte Carlo policy evaluation with epsilon-greedy
        iter_episode = 0
        current_state = self.grid_world.reset()
        state_trace   = [current_state]
        action_trace  = []
        reward_trace  = []
        seed = 1
        rng = np.random.default_rng(seed)
        
        loss_per_episode = []
        reward_per_episode = []

                
        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            # RUN Episode
            done = False
            while not done:
                current_state = self.grid_world.get_current_state()  # Get the current state
                action_probs = self.policy[current_state]  
                action = np.random.choice(self.action_space, p=action_probs) # random choose action based on all action probs
                next_state, reward, done = self.grid_world.step(action)
                state_trace.append(next_state) 
                action_trace.append(action)
                reward_trace.append(reward)
            # print(done)
            loss_per_episode.append(self.policy_evaluation(state_trace, action_trace, reward_trace))
            reward_per_episode.append(np.mean(reward_trace))
            self.policy_improvement()               
  
            # if iter_episode % 100 == 0:
            #     print(iter_episode)
            if iter_episode >= 10:
                lr = np.mean(reward_per_episode[-10:])
                loss = np.mean(loss_per_episode[-10:])
                # self.learning_curve.append(lr)
                # self.loss_curve.append(loss)
                wandb.log({"Episode": iter_episode, "lr": lr, "loss": loss})
            else:
                lr = np.mean(reward_per_episode)
                loss = np.mean(loss_per_episode)
                # self.learning_curve.append(lr)
                # self.loss_curve.append(loss)
                wandb.log({"Episode": iter_episode, "lr": lr, "loss": loss})

            state_trace   = [self.grid_world.get_current_state()]
            action_trace  = []
            reward_trace  = []
            iter_episode += 1

class SARSA(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for SARSA

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def policy_eval_improve(self, s, a, r, s2, a2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        # TODO: Evaluate Q value after one step and improve the policy
        loss = r + self.discount_factor*self.q_values[s2][a2]*(1-is_done) - self.q_values[s][a]
                                                       
        self.q_values[s][a] += self.lr*loss
        for a in range(self.action_space):
            if a == np.argmax(self.q_values[s]):
                self.policy[s][a] = self.epsilon/self.action_space + 1 - self.epsilon
            else:
                self.policy[s][a] = self.epsilon/self.action_space
        return abs(loss)
                              
        # raise NotImplementedError

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the TD policy evaluation with epsilon-greedy
        iter_episode = 0
        current_state = self.grid_world.reset()
        prev_s = None
        prev_a = None
        prev_r = None
        is_done = False
        
        loss_per_episode = []
        reward_per_episode = []
        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            reward_trace  = []
            loss_trace = []
            while True:
                current_state = self.grid_world.get_current_state()                
                action_probs = self.policy[current_state]
                action = np.random.choice(self.action_space, p=action_probs)
                next_state, reward, is_done = self.grid_world.step(action)
                next_action_probs = self.policy[next_state] 
                next_action = np.random.choice(self.action_space, p=next_action_probs)
                loss = self.policy_eval_improve(current_state, action, reward, next_state, next_action, is_done)
                loss_trace.append(loss)
                reward_trace.append(reward)
                if not is_done:
                    break
            loss_per_episode.append(np.mean(loss_trace))
            reward_per_episode.append(np.mean(reward_trace))
            iter_episode += 1
            # if iter_episode % 100 == 0:
                # print(iter_episode)
            if iter_episode >= 10:
                lr = np.mean(reward_per_episode[-10:])
                loss = np.mean(loss_per_episode[-10:])
                # self.learning_curve.append(lr)
                # self.loss_curve.append(loss)
                wandb.log({"Episode": iter_episode, "lr": lr, "loss": loss})
            else:
                lr = np.mean(reward_per_episode)
                loss = np.mean(loss_per_episode)
                # self.learning_curve.append(lr)
                # self.loss_curve.append(loss)
                wandb.log({"Episode": iter_episode, "lr": lr, "loss": loss})


class Q_Learning(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float, buffer_size: int, update_frequency: int, sample_batch_size: int):
        """Constructor for Q_Learning

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr                = learning_rate
        self.epsilon           = epsilon
        self.buffer            = deque(maxlen=buffer_size)
        self.update_frequency  = update_frequency
        self.sample_batch_size = sample_batch_size

    def add_buffer(self, s, a, r, s2, d) -> None:
        # TODO: add new transition to buffer
        self.buffer.append([s, a, r, s2, d])
        # raise NotImplementedError

    def sample_batch(self) -> np.ndarray:
        # TODO: sample a batch of index of transitions from the buffer
        index = np.random.randint(0, len(self.buffer)-1, self.sample_batch_size)
        batch = np.array([self.buffer[i] for i in index])
        return batch

        # raise NotImplementedError

    def policy_eval_improve(self, s, a, r, s2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        #TODO: Evaluate Q value after one step and improve the policy
        loss = r + self.discount_factor*max(self.q_values[s2])*(1-is_done) - self.q_values[s][a]
                                                       
        self.q_values[s][a] += self.lr*loss
        for a in range(self.action_space):
            if a == np.argmax(self.q_values[s]):
                self.policy[s][a] = self.epsilon/self.action_space + 1 - self.epsilon
            else:
                self.policy[s][a] = self.epsilon/self.action_space 
        return abs(loss)
        # raise NotImplementedError

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Q_Learning algorithm
        iter_episode = 0
        current_state = self.grid_world.reset()
        prev_s = None
        prev_a = None
        prev_r = None
        is_done = False
        transition_count = 0


        loss_per_episode = []
        reward_per_episode = []

        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here   
            loss_trace = []
            reward_trace = []    
            while True:
                current_state = self.grid_world.get_current_state()
                action_probs = self.policy[current_state]
                action = np.random.choice(self.action_space, p=action_probs)
                next_state, reward, is_done = self.grid_world.step(action)               
                self.add_buffer(current_state, action, reward, next_state, is_done)

                transition_count += 1
                
                if transition_count % self.update_frequency == 0:
                    batch = self.sample_batch()
                    for s, a, r, s2, d in batch:
                        loss = self.policy_eval_improve(int(s), int(a), r, int(s2), d)
                        loss_trace.append(loss)
                        reward_trace.append(r)
                if is_done:
                    break
                
            # print(loss_trace)
            # print(reward_trace)
            if not loss_trace and not reward_trace:
                loss_per_episode.append(0)
                reward_per_episode.append(0)
            else:
                loss_per_episode.append(np.mean(loss_trace))
                reward_per_episode.append(np.mean(reward_trace))
            
            # if iter_episode % 100 == 0:
                # print(iter_episode)
                if iter_episode >= 10:
                    lr = np.mean(reward_per_episode[-10:])
                    loss = np.mean(loss_per_episode[-10:])
                    # self.learning_curve.append(lr)
                    # self.loss_curve.append(loss)
                    wandb.log({"Episode": iter_episode, "lr": lr, "loss": loss})
                else:
                    lr = np.mean(reward_per_episode)
                    loss = np.mean(loss_per_episode)
                    # self.learning_curve.append(lr)
                    # self.loss_curve.append(loss)
                    wandb.log({"Episode": iter_episode, "lr": lr, "loss": loss})
            iter_episode += 1