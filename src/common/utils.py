import numpy as np
import torch

def unpack_batch(batch):
    """
    takes in a batch of experiences and returns numpy arrays for each field
    """
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))

    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


def calc_values_of_states(states, net, device="cpu"):
    
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)

def default_states_preprocessor(states):
    """
    Convert list of states into the form suitable for model. By default we assume Variable

    Args:
        states: list of numpy arrays with states

    Returns:
        cleaned variable
    """

    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    return torch.tensor(np_states)


def float32_preprocessor(states):
    """
    Convert list of states into the form suitable for model. By default we assume Variable

    Args:
        states: list of numpy arrays with states

    Returns:
        cleaned variable in the form of np.float32
    """

    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)
