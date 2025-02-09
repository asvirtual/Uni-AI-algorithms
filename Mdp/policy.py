import numpy as np


class Policy:
    def __init__(self, actions=None):
        self._actions = actions if actions is not None else {}


    def set_action(self, state, action):
        self._actions[state.get_id()] = action


    def get_action(self, state):
        return self._actions[state.get_id()]
    

    def print(self, actions_labels):
        for state_id, action in self._actions.items():
            print(f"State {state_id}: {actions_labels[action]}")


    def __getitem__(self, state):
        return self.get_action(state)
    

    def __str__(self):
        to_string = ""
        for state_id, action in self._actions.items():
            to_string += f"State {state_id}: {action}\n"

        return to_string
    

    def __eq__(self, other):
        return self._actions == other._actions