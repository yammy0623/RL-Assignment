import numpy as np

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
            v_new = np.zeros(self.grid_world.get_state_space()) # 不知為何v_new在裡面和在外面宣告會有差異
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
        self.policy = np.zeros((self.grid_world.get_state_space(), 4))

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        value = 0.0
        num_actions = self.grid_world.get_action_space()
        for action in range(num_actions):
            value += self.policy[state, action] * self.get_q_value(state, action)
        return value

        # raise NotImplementedError

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        while True:
            delta = 0.0  
            v_new = np.zeros(self.grid_world.get_state_space()) # 不知為何v_new在裡面和在外面宣告會有差異
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
            # print(self.policy[s])
            # k
            old_action = np.argmax(self.policy[s])
            action_values = np.zeros(self.grid_world.get_action_space())
            for a in range(self.grid_world.get_action_space()):
                action_values[a] = self.get_q_value(s, a)
            new_action = np.argmax(action_values)
            self.policy[s] = np.eye(self.grid_world.get_action_space())[new_action]
            # for a_index in range(self.grid_world.get_action_space()):
            #     if a_index == new_action:
            #         self.policy[s, a_index] = 1
            #     else:
            #         self.policy[s, a_index] = 0
            if old_action != new_action:
                policy_stable = False
        return policy_stable
        # raise NotImplementedError

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the policy iteration algorithm until convergence
        # Initialize policy (only action: 0 1 2 3) 不知道這個初始化會不會造成結果上的差異，可以寫在run內嗎?
        self.policy = np.ones((self.grid_world.get_state_space(), 4)) / 4
        while True:
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            if policy_stable:
                break
        self.policy = np.argmax(self.policy, axis=1)
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
        raise NotImplementedError

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        raise NotImplementedError

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        raise NotImplementedError

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the value iteration algorithm until convergence
        raise NotImplementedError


class AsyncDynamicProgramming(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the async dynamic programming algorithm until convergence
        raise NotImplementedError
