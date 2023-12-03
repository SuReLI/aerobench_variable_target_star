from abc import ABC, abstractmethod

import os
import imageio
import gym
try:
    import roboschool   # noqa: F401
except ModuleNotFoundError:
    pass

import torch

from commons.utils import NormalizedActions, ReplayMemory

from cfd.starccm.CFDcommunication import CFDcommunication

class AbstractAgent(ABC):

    def __init__(self, device, folder, config):

        self.folder = folder
        self.config = config
        self.device = device
        self.memory = ReplayMemory(self.config['MEMORY_CAPACITY'])

        #FLATPLATE/STARCCM/ELLIPSE
        if config["GAME"]["id"] == "STARCCMexternalfiles":
            self.eval_env = NormalizedActions(CFDcommunication(config))
        elif config["GAME"]["id"] == "flatplate":
            self.eval_env = NormalizedActions(FlatPlate(config))
        else:
            self.eval_env = NormalizedActions(gym.make(**self.config['GAME']))
            
        self.continuous = bool(self.eval_env.action_space.shape)

        self.state_size = self.eval_env.observation_space.shape[0]
        if self.continuous:
            self.action_size = self.eval_env.action_space.shape[0]
        else:
            self.action_size = self.eval_env.action_space.n

        self.display_available = 'DISPLAY' in os.environ

    @abstractmethod
    def select_action(self, state, episode=None, evaluation=False):
        pass

    def get_batch(self):

        transitions = self.memory.sample(self.config['BATCH_SIZE'])
        batch = list(zip(*transitions))

        # Divide memory into different tensors
        states = torch.FloatTensor(batch[0]).to(self.device)
        if self.continuous:
            actions = torch.FloatTensor(batch[1]).to(self.device)
        else:
            actions = torch.LongTensor(batch[1]).to(self.device)
        rewards = torch.FloatTensor(batch[2]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(batch[3]).to(self.device)
        done = torch.FloatTensor(batch[4]).unsqueeze(1).to(self.device)

        return states, actions, rewards, next_states, done

    @abstractmethod
    def optimize(self):
        pass

    def evaluate(self, n_ep=1, render=False, gif=False, test=False, appli=False):
        rewards = []
        if gif:
            writer = imageio.get_writer(self.folder + '/results.gif', duration=0.005)
        render = render and self.display_available

        try:
            for i in range(n_ep):
                state = self.eval_env.reset(Btype=self.config['BTYPE_EVAL'])
                reward = 0
                done = False
                steps = 0
                while not done and steps < self.config['MAX_STEPS']:
                    action = self.select_action(state, evaluation=True)
                    if self.config["GAME"]["id"] == "STARCCMexternalfiles":
                        self.eval_env.finishCFD()        
                    state, r, done, _ = self.eval_env.step(action)
                    if render:
                        self.eval_env.render()
                    if i == 0 and gif:
                        writer.append_data(self.eval_env.render(mode='rgb_array'))
                    reward += r
                    if self.config["GAME"]["id"] == "STARCCMexternalfiles":
                        #set as done if the number of maximum steps is reached even if not
                        #reached the final position to avoid the simulation to continue
                        if not done and steps == self.config["MAX_STEPS"] - 1:
                            done = True                           
                    steps += 1
                rewards.append(reward)
             
                if self.config["GAME"]["id"] == "flatplate" or self.config["GAME"]["id"] == "STARCCMexternalfiles":
                    self.eval_env.print_won_or_lost(state)

                if test and appli:
                    # SAVE variables at the end of episode
                    self.eval_env.fill_array_tobesaved()

        except KeyboardInterrupt:
            if not render:
                raise

        finally:
            #Sand
            if test:
                testfolder = f'{self.folder}/test'
                # Create test folder
                if not os.path.exists(testfolder):
                    os.makedirs(testfolder)
                # DUMP variables at the end of episode
                if appli:
                    self.eval_env.print_array_in_files(testfolder)
                    self.eval_env.plot_testing_output(rewards, testfolder)

            self.eval_env.close()

            if gif:
                print(f"Saved gif in {self.folder+'/results.gif'}")
                writer.close()

            if test and self.config["GAME"]["id"] == "STARCCMexternalfiles":
                #end simulation of STARCCM+
                self.eval_env.finishCFD(True)

        score = sum(rewards)/len(rewards) if rewards else 0
        return rewards

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self, folder=None):
        pass
