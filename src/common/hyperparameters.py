PARAMS = {
    "banana": {
        "env_name": "Banana.app",
        "env_type": "unity",
        "stop_reward": 200,
        "run_name": "banana",
        "replay_size": 100000,
        "replay_initial": 1000,
        "target_net_sync": 100,
        "epsilon_frames": 10 ** 4,
        "epsilon_start": 1.0,
        "epsilon_final": 0.1,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "batch_size": 64,
    },
    "cartpole": {
        "env_name": "CartPole-v0",
        "env_type": "basic",
        "num_env": 5,
        "stop_reward": 195,
        "run_name": "cartpole",
        "replay_size": 100000,
        "replay_initial": 1000,
        "target_net_sync": 100,
        "epsilon_frames": 10 ** 4,
        "epsilon_start": 1.0,
        "epsilon_final": 0.1,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "batch_size": 64,
        "step_count": 10,
        "grad_clip": 5,
        "beta": 0.01,
    },
    "cartpole_a3c": {
        "env_name": "CartPole-v0",
        "env_type": "basic",
        "num_env": 2,
        "num_procs": 4,
        "stop_reward": 195,
        "run_name": "cartpole",
        "replay_size": 100000,
        "replay_initial": 1000,
        "target_net_sync": 100,
        "epsilon_frames": 10 ** 4,
        "epsilon_start": 1.0,
        "epsilon_final": 0.1,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "batch_size": 64,
        "step_count": 4,
        "grad_clip": 5,
        "beta": 0.01,
    },
    "lunarlander": {
        "env_name": "LunarLander-v2",
        "env_type": "basic",
        "stop_reward": 200.0,
        "run_name": "lunarlander",
        "replay_size": 100000,
        "replay_initial": 1000,
        "target_net_sync": 100,
        "epsilon_frames": 10 ** 5,
        "epsilon_start": 1.0,
        "epsilon_final": 0.01,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "batch_size": 64,
    },
    "pong": {
        "env_name": "PongNoFrameskip-v4",
        "env_type": "atari",
        "stop_reward": 18.0,
        "run_name": "pong",
        "replay_size": 100000,
        "replay_initial": 10000,
        "target_net_sync": 1000,
        "epsilon_frames": 10 ** 5,
        "epsilon_start": 1.0,
        "epsilon_final": 0.02,
        "learning_rate": 0.0001,
        "gamma": 0.99,
        "batch_size": 32,
    },
    "breakout-small": {
        "env_name": "BreakoutNoFrameskip-v4",
        "env_type": "atari",
        "stop_reward": 500.0,
        "run_name": "breakout-small",
        "replay_size": 3 * 10 ** 5,
        "replay_initial": 20000,
        "target_net_sync": 1000,
        "epsilon_frames": 10 ** 6,
        "epsilon_start": 1.0,
        "epsilon_final": 0.1,
        "learning_rate": 0.0001,
        "gamma": 0.99,
        "batch_size": 64,
    },
    "breakout": {
        "env_name": "BreakoutNoFrameskip-v4",
        "env_type": "atari",
        "stop_reward": 500.0,
        "run_name": "breakout",
        "replay_size": 10 ** 6,
        "replay_initial": 50000,
        "target_net_sync": 10000,
        "epsilon_frames": 10 ** 6,
        "epsilon_start": 1.0,
        "epsilon_final": 0.1,
        "learning_rate": 0.00025,
        "gamma": 0.99,
        "batch_size": 32,
    },
    "invaders": {
        "env_name": "SpaceInvadersNoFrameskip-v4",
        "env_type": "atari",
        "stop_reward": 500.0,
        "run_name": "breakout",
        "replay_size": 10 ** 6,
        "replay_initial": 50000,
        "target_net_sync": 10000,
        "epsilon_frames": 10 ** 6,
        "epsilon_start": 1.0,
        "epsilon_final": 0.1,
        "learning_rate": 0.00025,
        "gamma": 0.99,
        "batch_size": 32,
    },
    "pong_a2c": {
        "env_name": "PongNoFrameskip-v4",
        "env_type": "atari",
        "num_env": 1,
        "stop_reward": 18.0,
        "run_name": "pong",
        "replay_size": 100000,
        "replay_initial": 10000,
        "target_net_sync": 1000,
        "epsilon_frames": 10 ** 5,
        "epsilon_start": 1.0,
        "epsilon_final": 0.02,
        "learning_rate": 0.0001,
        "beta": 0.01,
        "num_envs": 50,
        "step_count": 4,
        "clip_grad": 0.1,
        "gamma": 0.99,
        "batch_size": 128,
    },
}
