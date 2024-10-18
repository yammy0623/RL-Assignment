import numpy as np
import heapq 

from gridworld import GridWorld


class DynamicProgramming:
    """Base class for dynamic programming algorithms"""

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for DynamicProgramming

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.threshold = 1e-4  # default threshold for convergence
        self.values = np.zeros(grid_world.get_state_space())  # V(s)
        self.policy = np.zeros(grid_world.get_state_space(), dtype=int)  # pi(s) 22

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold for convergence

        Args:
            threshold (float): threshold for convergence
        """
        self.threshold = threshold

    def get_policy(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy
        """
        return self.policy

    def get_values(self) -> np.ndarray:
        """Return the values

        Returns:
            np.ndarray: values
        """
        return self.values

    def get_q_value(self, state: int, action: int) -> float:
        """Get the q-value for a state and action

        Args:
            state (int)
            action (int)

        Returns:
            float
        """
        # TODO: Get reward from the environment and calculate the q-value
        next_state, reward, done = self.grid_world.step(state, action)
        if not done:
            reward += self.discount_factor * self.values[next_state] 
        return reward


class IterativePolicyEvaluation(DynamicProgramming):
    def __init__(
        self, grid_world: GridWorld, policy: np.ndarray, discount_factor: float
    ):
        """Constructor for IterativePolicyEvaluation

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): policy (probability distribution state_spacex4)
            discount (float): discount factor gamma
        """
        super().__init__(grid_world, discount_factor)
        self.policy = policy

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float: value
        """
        # TODO: Get the value for a state by calculating the q-values
        # policy: [22, 4]
        value = 0.0
        num_actions = self.grid_world.get_action_space()
        for action in range(num_actions):
            value += self.policy[state, action] * self.get_q_value(state, action)
        return value
        # raise NotImplementedError

    def evaluate(self):
        """Evaluate the policy and update the values for one iteration"""
        # TODO: Implement the policy evaluation step 
        while True:
            delta = 0.0  
            v_new = np.zeros(self.grid_world.get_state_space()) 
            for s in range(self.grid_world.get_state_space()):
                v = self.values[s]   
                v_new[s] = self.get_state_value(s) 
                # self.values[s] = self.get_state_value(s)
                delta = max(delta, abs(v - v_new[s]))
            self.values = v_new # State should be updated after the iteration
            if delta < self.threshold:
                break            
        # raise NotImplementedError

    def run(self) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the iterative policy evaluation algorithm until convergence
        self.evaluate()
        # raise NotImplementedError

class PolicyIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for PolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values        
        return self.get_q_value(state, self.policy[state])
        # raise NotImplementedError

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        while True:
            delta = 0.0  
            v_new = np.zeros(self.grid_world.get_state_space()) 
            for s in range(self.grid_world.get_state_space()):
                v = self.values[s]   
                v_new[s] = self.get_state_value(s) 
                delta = max(delta, abs(v - v_new[s]))
            self.values = v_new # State should be updated after the iteration
            if delta < self.threshold:
                break   
        # raise NotImplementedError

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        policy_stable = True
        for s in range(self.grid_world.get_state_space()):
            old_action = self.policy[s]
            action_values = np.zeros(self.grid_world.get_action_space())
            for a in range(self.grid_world.get_action_space()):
                action_values[a] = self.get_q_value(s, a)
            new_action = np.argmax(action_values)
            self.policy[s] = new_action
            if old_action != new_action:
                policy_stable = False
        return policy_stable
        # raise NotImplementedError

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the policy iteration algorithm until convergence
        # 這邊的policy是 deterministic policy: 所以policy只有一維，也就是固定好每個state的action是甚麼了
        time = 1
        while True:
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            # print(time)
            if policy_stable:
                break
            time += 1
        # raise NotImplementedError


class ValueIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        action_value = np.zeros(self.grid_world.get_action_space())
        for a in range(self.grid_world.get_action_space()):
            action_value[a] = self.get_q_value(state, a)   
        return action_value       
        # raise NotImplementedError

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        while True:
            delta = 0.0  
            v_new = np.zeros(self.grid_world.get_state_space())
            for s in range(self.grid_world.get_state_space()):
                v = self.values[s]
                v_new[s] = max(self.get_state_value(s))
                delta = max(delta, abs(v - v_new[s]))
            self.values = v_new # State should be updated after the iteration
            if delta < self.threshold:
                break 
        # raise NotImplementedError

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        # Value iteration不需要iterate policy
        for s in range(self.grid_world.get_state_space()):
            new_action = np.argmax(self.get_state_value(s))
            self.policy[s] = new_action
        # raise NotImplementedError

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the value iteration algorithm until convergence
        self.policy_evaluation()
        self.policy_improvement()

class AsyncDynamicProgramming(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)
    
    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        action_value = np.zeros(self.grid_world.get_action_space())
        for a in range(self.grid_world.get_action_space()):
            action_value[a] = self.get_q_value(state, a)   
        return action_value   

    
    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        while True:
            delta = 0.0  
            for s in range(self.grid_world.get_state_space()):
                v = self.values[s]
                self.values[s] = max(self.get_state_value(s))
                delta = max(delta, abs(v - self.values[s]))
            if delta < self.threshold:
                break 
        # raise NotImplementedError

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        policy_stable = True
        for s in range(self.grid_world.get_state_space()):
            old_action = self.policy[s]
            new_action = np.argmax(self.get_state_value(s))
            self.policy[s] = new_action
            if old_action != new_action:
                policy_stable = False
        return policy_stable
        # raise NotImplementedError

    def in_place_DP(self) -> None:
        self.policy_evaluation()
        self.policy_improvement()
            
    def prioritized_sweeping(self): 
        PQueue = [] # priority queue
        bellman_err = np.zeros(self.grid_world.get_state_space()) # Store error
        for s in range(self.grid_world.get_state_space()):
            v = self.values[s]
            self.values[s] = max(self.get_state_value(s))
            bellman_err[s] = abs(v - self.values[s])
        largest_err_s = np.argmax(bellman_err)
        heapq.heappush(PQueue, (-bellman_err[largest_err_s], largest_err_s))

        delta = 0.0
        while PQueue:
            p, s = heapq.heappop(PQueue)
            v = self.values[s]
            self.values[s] = max(self.get_state_value(s))
            delta = max(delta, abs(v - self.values[s]))            
            if delta < self.threshold:
                break 
            for s in range(self.grid_world.get_state_space()):
                v = self.values[s]
                self.values[s] = max(self.get_state_value(s))
                bellman_err[s] = abs(v - self.values[s])
            largest_err_s = np.argmax(bellman_err)
            if bellman_err[largest_err_s] > self.threshold:
                heapq.heappush(PQueue, (-bellman_err[largest_err_s], largest_err_s))
        self.policy_improvement()
    
    
    def realtime_DP(self):
        # 只對於探索過的狀態進行更新
        # Initialize a set to keep track of visited states
        while True:     
            for s in range(self.grid_world.get_state_space()):
                visited_states = []
                while True:
                    delta = 0.0  
                    v = self.values[s]
                    self.values[s] = max(self.get_state_value(s))
                    next_action = np.argmax(self.get_state_value(s))
                    self.policy[s] = next_action 
                    visited_states.append([s,self.values[s]] )
                    next_state, reward, done = self.grid_world.step(s, next_action)
                    if done:
                        break
                    s = next_state
                delta = max(delta, abs(v - self.values[s]))
                if delta < self.threshold:
                    break   
            if self.policy_improvement():
                break      

    def rtdp_prioritized_sweeping(self):
        PQueue = [] # priority queue
        bellman_err = np.zeros(self.grid_world.get_state_space()) # Store error
        for s in range(self.grid_world.get_state_space()):
            v = self.values[s]
            self.values[s] = max(self.get_state_value(s))
            bellman_err[s] = abs(v - self.values[s])
        largest_err_s = np.argmax(bellman_err)
        heapq.heappush(PQueue, (-bellman_err[largest_err_s], largest_err_s))

        delta = 0.0
        while PQueue:
            p, s = heapq.heappop(PQueue)
            v = self.values[s]
            self.values[s] = max(self.get_state_value(s))
            delta = max(delta, abs(v - self.values[s]))            
            if delta < self.threshold:
                break
            for s in range(self.grid_world.get_state_space()):
                while True:
                    delta = 0.0  
                    v = self.values[s]
                    self.values[s] = max(self.get_state_value(s))
                    next_action = np.argmax(self.get_state_value(s))
                    self.policy[s] = next_action 
                    bellman_err[s] = abs(v - self.values[s])
                    largest_err_s = np.argmax(bellman_err)
                    if bellman_err[largest_err_s] > self.threshold:
                        heapq.heappush(PQueue, (-bellman_err[largest_err_s], largest_err_s))

                    next_state, reward, done = self.grid_world.step(s, next_action)
                    if done:
                        break
                    s = next_state
            delta = max(delta, abs(v - self.values[s]))
            if delta < self.threshold:
                break   
            if self.policy_improvement():
                break   

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the async dynamic programming algorithm until convergence
        # self.in_place_DP()
        self.prioritized_sweeping()
        # self.realtime_DP()
        # self.rtdp_prioritized_sweeping()
        # raise NotImplementedError
