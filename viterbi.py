import numpy as np

'''
Function to calculate the viterbi probabilities for a single instant, given that the state can be described with a single boolean random variable
emission: the registered emission (i.e. c or d)
emission_probs: the emission probability distribution
transition_probs: the transition probability distribution from previous to current state
parent_probs: the probability distribution for the previous instant (result of another call of the same funtion or given for t = 0)
'''
def calculate_viterbi_state(emission, emission_probs, transition_probs=np.array([[1,1],[1,1]]), parent_probs=np.array([0.5, 0.5])):
    emission_prob_true = emission_probs[0][emission]
    emission_prob_false = emission_probs[1][emission]

    # true -> true, false -> true
    transition_true_candidates = [transition_probs[0, 0] * parent_probs[0], transition_probs[1, 0] * parent_probs[1]]
    # true -> false, false -> false
    transition_false_candidates = [transition_probs[0, 1] * parent_probs[0], transition_probs[1, 1] * parent_probs[1]]

    return np.array([
        emission_prob_true * np.max(transition_true_candidates),
        emission_prob_false * np.max(transition_false_candidates)
    ])


def calculate_viterbi_path(emissions, emission_probs, transition_probs, float_precision=10):
    states = [calculate_viterbi_state(emissions[0], emission_probs, transition_probs)] # first state (instant t1) is just a transition from an initial instant t0 that has both states equally probable
    
    for idx, emission in enumerate(emissions[1:]):
        states.append(
            calculate_viterbi_state(emission, emission_probs, transition_probs, parent_probs=states[idx])
        )

    return [{ 
        "state": "true" if state[0] > state[1] else "false",
        'P["true"]': round(state[0], float_precision),
        'P["false"]': round(state[1], float_precision),
    } for state in states]

# '''
# Problem data
transition_probs = np.array([
    [0.5, 0.5], # clean -> clean, clean -> dirty
    [0.8, 0.2]  # dirty -> clean, dirty -> dirty
])

emission_probs = np.array([
    [0.4, 0.4, 0.1, 0.1], # clean
    [0.1, 0.1, 0.4, 0.4]  # dirty
]) # ma, pa, pe, po

# indexes
ma, pa, pe, po = 0, 1, 2, 3
c, d = 0, 1

emissions = [ma, ma, pe, pe]
# '''
'''
# Problem data
transition_probs = np.array([
    [0.7, 0.3], # rain -> rain, rain -> dry
    [0.3, 0.7]  # dry -> rain, dry -> dry
])

emission_probs = np.array([
    [0.9, 0.1], # rain
    [0.2, 0.8]  # dry
]) # umbrella, no umbrella

# indexes
umbrella, no_umbrella = 0, 1
rain, dry = 0, 1

emissions = [umbrella, umbrella, no_umbrella, umbrella, umbrella]
'''

path = calculate_viterbi_path(emissions, emission_probs, transition_probs, float_precision=6)
for node in path: print(node)