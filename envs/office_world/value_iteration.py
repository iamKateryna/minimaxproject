def value_iteration(states, actions, labelling_fuction, transitions, reward_machine,gamma):
    """
    Standard value iteration to compute optimal policies for the grid environments.
    
    PARAMS
    ----------
    states:     List of states
    actions:     List of actions
    labelling_fuction:     Labeling function (it is a dictionary from states to events)
    transitions:     Transitions (it is a dictionary from SxA -> S)
    reward_machine:    Reward machine
    gamma: Discount factor 

    RETURNS
    ----------
    Optimal deterministic policy (dictionary maping from states (SxU) to actions)
    """
    U = reward_machine.get_states() # RM states
    V = dict([((state,rm_state),0) for state in states for rm_state in U])
    V_error = 1

    # Computing the optimal value function
    while V_error > 0.0000001:
        V_error = 0
        for state in states:
            for u in U:
                q_values = []
                for action in actions:
                    next_state = transitions[(state, action)]
                    l  = '' if next_state not in labelling_fuction else labelling_fuction[next_state]
                    next_u, reward, done = reward_machine.step(u, l, None)
                    if done: q_values.append(reward)
                    else:    q_values.append(reward+gamma * V[(next_state, next_u)])
                v_new = max(q_values)
                V_error = max([V_error, abs(v_new-V[(state,u)])])
                V[(state,u)] = v_new

    # Extracting the optimal policy
    policy = {}
    for state in states:
        for u in U:
            q_values = []
            for action in actions:
                next_state = transitions[(state, action)]
                l  = '' if next_state not in labelling_fuction else labelling_fuction[next_state]
                next_u, reward, done = reward_machine.step(u, l, None)
                if done: q_values.append(reward)
                else:    q_values.append(reward+gamma * V[(next_state, next_u)])
            a_i = max((x,i) for i,x in enumerate(q_values))[1] # argmax over the q-valies
            policy[(state, u)] = actions[a_i]

    return policy

