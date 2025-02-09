import numpy as np
from state import State
from problem import Problem


actions = {
    0: 'up',
    1: 'down',
    2: 'left',
    3: 'right'
}

actions_probs = np.array([
    np.array([0.8, 0.0, 0.1, 0.1]), # up
    np.array([0.0, 0.8, 0.1, 0.1]), # down
    np.array([0.1, 0.1, 0.8, 0.0]), # left
    np.array([0.1, 0.1, 0.0, 0.8])  # right
])

states = [
    State(
        0,
        actions_probs,
        -0.04,
        [0, 4, 0, 1]
    ),
    State(
        1,
        actions_probs,
        -0.04,
        [1, 1, 0, 2]
    ),
    State(
        2,
        actions_probs,
        -0.04,
        [2, 5, 1, 3]
    ),
    State(
        3,
        None,
        1,
        None,
        terminal=True
    ),
    State(
        4,
        actions_probs,
        -0.04,
        [0, 7, 4, 4]
    ),
    State(
        5,
        actions_probs,
        -0.04,
        [2, 9, 5, 6]
    ),
    State(
        6,
        None,
        -1,
        None,
        terminal=True
    ),
    State(
        7,
        actions_probs,
        -0.04,
        [4, 7, 7, 8],
        initial=True
    ),
    State(
        8,
        actions_probs,
        -0.04,
        [8, 8, 7, 9]
    ),
    State(
        9,
        actions_probs,
        -0.04,
        [5, 9, 8, 10]
    ),
    State(
        10,
        actions_probs,
        -0.04,
        [6, 10, 9, 10]
    )
]

problem = Problem(
    states,
    actions,
    actions_probs,
    # gamma=0
)

policy = problem.policy_iteration()
for state in states:
    if state.is_terminal():
        continue

print("Policy using policy iteration:")
policy.print(actions)

utilities, iterations = problem.value_iteration()
print(f"\nUtilities using value iteration, after {iterations} iterations:")
print(utilities)
print("\nPolicy using value iteration:")
policy = problem.get_policy(utilities)
policy.print(actions)