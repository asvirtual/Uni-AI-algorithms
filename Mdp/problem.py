import numpy as np
from state import State
from policy import Policy


class Problem:
    def __init__(self, states, actions, transition_model, gamma=1):
        if not all(isinstance(state, State) for state in states):
            raise ValueError("[ERROR] All states must be of type State")

        self._states = states
        self._actions = actions
        self._transition_model = transition_model
        self._gamma = gamma


    def compute_policy_utilities(self, policy, old_utilities=None):
        if old_utilities is None:
            old_utilities = np.zeros(len(self._states))

        non_terminal_states = [state for state in self._states if not state.is_terminal()]
        new_utilities_system = np.zeros((len(self._states), len(self._states)))
        new_utilities_system_constants = np.zeros(len(self._states))

        # Initialize the system of equations for the utilities
        for state in non_terminal_states:
            new_utilities_system[state.get_id()][state.get_id()] = 1

        for state in non_terminal_states:
            # Get the action from the policy
            action = policy.get_action(state)

            # Compute the utility for the state, in form of a numpy array that will be fed to the np.linalg.solve function
            # to solve the linear system of equations defining the utilities
            for actual_action in self._actions:
                new_state_id = state.get_new_state(actual_action)
                new_state = self._states[new_state_id]
                
                # If the state is terminal, the utility is the reward, and there is no need to add it to the system of equations
                # since it is a constant, so we add it to the constants vector
                if new_state.is_terminal(): 
                    new_utilities_system_constants[state.get_id()] += new_state.get_reward() * self._gamma * self._transition_model[action][actual_action]
                    continue
                
                new_utilities_system[state.get_id()][new_state_id] -= self._transition_model[action][actual_action] * self._gamma
                new_utilities_system_constants[state.get_id()] += self._transition_model[action][actual_action] * self._states[new_state_id].get_reward()

        # Remove the zero rows and columns from the system of equations and the corresponding constants derived from terminal states
        new_utilities_system = np.delete(new_utilities_system, [state.get_id() for state in self._states if state.is_terminal()], axis=0)
        new_utilities_system = np.delete(new_utilities_system, [state.get_id() for state in self._states if state.is_terminal()], axis=1)
        new_utilities_system_constants = np.delete(new_utilities_system_constants, [state.get_id() for state in self._states if state.is_terminal()], axis=0)

        utilities = np.linalg.solve(new_utilities_system, new_utilities_system_constants)

        # Fill the utilities array with the utilities of the terminal states, which are equal to their rewards
        for state in self._states:
            if state.is_terminal():
                utilities = np.insert(utilities, state.get_id(), state.get_reward())

        return utilities
    

    def improve_policy(self, utilities):
        new_policy = Policy()

        for state in self._states:
            if state.is_terminal(): continue

            best_action = None
            best_utility = -np.inf

            for action in self._actions:
                new_state_id = state.get_new_state(action)
                utility = utilities[new_state_id]

                if utility > best_utility:
                    best_utility = utility
                    best_action = action

            new_policy.set_action(state, best_action)

        return new_policy


    def policy_iteration(self, threshold=1e-3):
        policy = Policy()
        for state in self._states:
            policy.set_action(state, 2)

        old_utilities = np.full(len(self._states), -np.inf)
        while True:
            utilities = self.compute_policy_utilities(policy)
            new_policy = self.improve_policy(utilities)

            if np.linalg.norm(utilities - old_utilities) < threshold:
                break

            policy = new_policy
            old_utilities = utilities

        return policy
    

    def compute_utilities(self, utilities):
        new_utilities = np.zeros(len(self._states))
        # Compute the utilities for each state
        for state in self._states:
            # If the state is terminal, the utility is the reward
            if state.is_terminal():
                new_utilities[state.get_id()] = state.get_reward()
                continue
            
            # Otherwise, compute the utility using the Bellman equation
            best_utility = -np.inf
            # Iterate over the actions to find the best utility
            for action in self._actions:
                new_utility = 0
                # Iterate over the neighbor states to compute the utility, which is the sum of the reward and the discounted utility times the transition probability
                for action_idx, new_state_id in enumerate(state.get_neighbor_states()):
                    # If the state is terminal, the utility is the reward (times the transition probability) without the discounted utility
                    if self._states[new_state_id].is_terminal():
                        new_utility += self._transition_model[action][action_idx] * (self._states[new_state_id].get_reward())
                    # Otherwise, the utility is the reward plus the discounted utility (times the transition probability)
                    else:
                        new_utility += self._transition_model[action][action_idx] * (self._states[new_state_id].get_reward() + self._gamma * utilities[new_state_id])

                if new_utility > best_utility:
                    best_utility = new_utility

            # Update the utility for the state
            new_utilities[state.get_id()] = best_utility

        return new_utilities


    def value_iteration(self, max_iterations=100, threshold=1e-3):
        # Initialize the utilities with random values, except for the terminal states, which have utilities equal to their rewards
        old_utilities = np.random.rand(len(self._states))
        for state in self._states:
            if state.is_terminal():
                old_utilities[state.get_id()] = state.get_reward()

        utilities = None
        count = 0
        for _ in range(max_iterations):
            # Compute the utilities for each state
            utilities = self.compute_utilities(old_utilities)

            # If the difference between the new utilities and the old utilities is less than the threshold, stop the iteration
            if np.linalg.norm(utilities - old_utilities) < threshold:
                break

            old_utilities = utilities
            count += 1

        return utilities, count
    

    def get_policy(self, utilities):
        policy = Policy()
        # Compute the policy using the utilities
        for state in self._states:
            if state.is_terminal(): continue

            best_action = None
            best_utility = -np.inf

            # Iterate over the actions to find the best action
            for action in self._actions:
                new_state_id = state.get_new_state(action)
                utility = utilities[new_state_id]

                if utility > best_utility:
                    best_utility = utility
                    best_action = action

            # Set the best action for the state
            policy.set_action(state, best_action)

        return policy