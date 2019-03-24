from common import utils


def evaluate_a2c(
    net, env, count=10, device="cpu", preprocessor=utils.default_states_preprocessor
):
    """
	Run several episodes to test the performance of the a2c agent

	Args:
		net: network
		env: environment 
		device: decice used to compute graph
		preprocessor: preprocessing method

	Returns:
		average reward
		average steps
	"""
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            action_v = net(obs_v)[0]
            action = action_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count
