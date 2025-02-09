class State:
    def __init__(self, state_id, transition_table, reward, neighbor_states, terminal=False, initial=False):
        self._state_id = state_id
        self._transition_table = transition_table
        self._reward = reward
        self._neighbor_states = neighbor_states
        self._terminal = terminal
        self._initial = initial

    
    def get_id(self):
        return self._state_id
    

    def get_new_state(self, action):
        return self._neighbor_states[action]
    

    def get_reward(self):
        return self._reward
    

    def is_terminal(self):
        return self._terminal
    

    def get_neighbor_states(self):
        return self._neighbor_states.copy()
    