# code partly from https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/vec_env/subproc_vec_env.py

from datetime import datetime
import numpy as np
import multiprocessing
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper

import torch

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            
            # if done:
            #     ob = env.reset()

            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class SubprocVecEnvTorch(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        self.nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        # using batch size larger than 1 (more than 1 environment) seems to slow things down
        # but can still use high policy batch size

        time_start = datetime.now()
        results = [remote.recv() for remote in self.remotes]
        time_end = datetime.now()
        # print(f'step_wait: waited {"%.2f" % (time_end - time_start).total_seconds()}s for results')
        self.waiting = False
        obs, rews, dones, infos = map(list, zip(*results))
        # eliminate all "zero" observations (returned if an environment has terminated)
        longest_shape_len = 0
        longest_tensor = None

        for obs_idx in range(len(obs)):
            if len(obs[obs_idx].shape) > longest_shape_len:
                longest_tensor = obs[obs_idx]
                longest_shape_len = len(longest_tensor.shape)
        
        for obs_idx in range(len(obs)):
            if len(obs[obs_idx].shape) < longest_shape_len:
                obs[obs_idx] = torch.zeros_like(longest_tensor)

        return torch.stack(obs), torch.stack(rews), torch.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return torch.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return torch.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
