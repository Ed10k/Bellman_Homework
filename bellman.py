# imports
import numpy as np

# action_set = set

states = ['A', 'B', 'C', 'D']
actions = [1, 2]
rewards = {
    ('A', 1): -10,
    ('A', 2): -10,
    ('B', 1): -10,
    ('B', 2): -10,
    ('C', 1): -10,
    ('C', 2): -10,
    ('D', 1): 100,  # goal state, +100 for the reward
    ('D', 2): 100,  # goal state, +100 for the reward

}

# The different policies and their action probabilities
policy = {
    'policy1': {
        ('A', 1): 1, ('A', 2): 0,
        ('B', 1): 1, ('B', 2): 0,
        ('C', 1): 1, ('C', 2): 0},

    'policy2': {
        ('A', 1): 0, ('A', 2): 1,
        ('B', 1): 0, ('B', 2): 1,
        ('C', 1): 0, ('C', 2): 1},

    'policy3': {
        ('A', 1): .4, ('A', 2): .6,
        ('B', 1): 1, ('B', 2): 0,
        ('C', 1): 0, ('C', 2): 1}
}

state_value_table = {'A': 0,
                     'B': 0,
                     'C': 0,
                     'D': 0}

state_transition_model = {
    # Transitions from state A
    ('A', 1, 'A'): 0,
    ('A', 1, 'B'): 0,
    ('A', 1, 'C'): 1,
    ('A', 1, 'D'): 0,
    ('A', 2, 'A'): 0.1,
    ('A', 2, 'B'): 0.9,
    ('A', 2, 'C'): 0,
    ('A', 2, 'D'): 0,

    # Transitions from state B
    ('B', 1, 'A'): 0.9,
    ('B', 1, 'B'): 0.1,
    ('B', 1, 'C'): 0,
    ('B', 1, 'D'): 0,
    ('B', 2, 'A'): 0,
    ('B', 2, 'B'): 1,
    ('B', 2, 'C'): 0,
    ('B', 2, 'D'): 0,

    # Transitions from state C
    ('C', 1, 'A'): 0.1,
    ('C', 1, 'B'): 0,
    ('C', 1, 'C'): 0,
    ('C', 1, 'D'): 0.9,
    ('C', 2, 'A'): 0,
    ('C', 2, 'B'): 0,
    ('C', 2, 'C'): 0,
    ('C', 2, 'D'): 1,

}


def evaluate_policy(specific_policy, V, state_transition_model, rewards):
    iterations = 100
    gamma = 0.8
    for i in range(iterations):
        for state in states:
            if state != 'D':
                temp_value = 0  # Initialize the temporary value to 0
                for action in actions:
                    temp_value += (specific_policy[(state, action)] *
                                   sum(state_transition_model[(state, action, next_state)] *
                                       (rewards[(state, action)] + gamma * V[next_state])
                                       for next_state in states
                                       )

                                   )
                V[state] = temp_value  # Update the state value after considering all actions
    return V


# main

def main():
    v_initial = {'A': 0, 'B': 0, 'C': 0, 'D': 1003}
    v_policy1 = evaluate_policy(policy['policy1'], v_initial.copy(), state_transition_model, rewards)
    print(v_policy1)

    v_policy2 = evaluate_policy(policy['policy2'], v_initial.copy(), state_transition_model, rewards)
    print(v_policy2)

    v_policy3 = evaluate_policy(policy['policy3'], v_initial.copy(), state_transition_model, rewards)
    print(v_policy3)


main()
