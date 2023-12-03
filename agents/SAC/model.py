import numpy as np

import torch
import torch.nn.functional as F

from commons.plotter import Plotter
from commons.Abstract_Agent import AbstractAgent
from commons.networks import Critic
from commons.network_modules import SoftActorNetwork


class SAC(AbstractAgent):

    def __init__(self, device, folder, config):
        super().__init__(device, folder, config)

        self.critic_A = Critic(self.state_size, self.action_size, device, config)
        self.critic_B = Critic(self.state_size, self.action_size, device, config)
        self.soft_actor = SoftActorNetwork(self.state_size, self.action_size, self.config['HIDDEN_PI_LAYERS'], device).to(device)
        self.soft_actor_optimizer = torch.optim.Adam(self.soft_actor.parameters(), lr=self.config['ACTOR_LR'])


        if self.config['AUTO_ALPHA']:
            self.target_entropy = -np.prod(self.eval_env.action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config['ALPHA_LR'])
            self.alpha = torch.exp(self.log_alpha)
        else: 
            self.alpha = 0.2

        self.plotter = Plotter(config, device, folder)

    def select_action(self, state, episode=None, evaluation=False):
        assert (episode is not None) or evaluation
        return self.soft_actor.select_action(state, evaluation)

    def optimize(self):

        if len(self.memory) < self.config['BATCH_SIZE']:
            return {}

        states, actions, rewards, next_states, done = self.get_batch()

        current_Q1 = self.critic_A(states, actions)
        current_Q2 = self.critic_B(states, actions)
        next_actions, log_prob = self.soft_actor.evaluate(states)
    
        # Compute next state values at t+1 using target critic network
        target_Qa = self.critic_A.target(next_states, next_actions).detach()
        target_Qb = self.critic_B.target(next_states, next_actions).detach()
        target_Q = torch.min(target_Qa, target_Qb)
        target_Q = rewards + (1 - done) * self.config['GAMMA'] * (target_Q - self.alpha * log_prob)

        critic1_loss = F.mse_loss(current_Q1, target_Q.detach())
        critic2_loss = F.mse_loss(current_Q2, target_Q.detach())
        self.critic_A.update(critic1_loss)
        self.critic_B.update(critic2_loss)

        # Compute the next value of alpha
        if self.config['AUTO_ALPHA']:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().detach()
        else:
            self.alpha = 0.2

        loss_actor = (self.alpha * log_prob - self.critic_A(states, next_actions)).mean()
        
        self.soft_actor_optimizer.zero_grad()
        loss_actor.backward()
        self.soft_actor_optimizer.step()

        self.critic_A.update_target(self.config['TAU'])
        self.critic_B.update_target(self.config['TAU'])

        return {'Q1_loss': critic1_loss.item(), 'Q2_loss': critic2_loss.item(), 'actor_loss': loss_actor.item()}

    def save(self):
        print("\033[91m\033[1mModel saved in", self.folder, "\033[0m")
        self.critic_A.save(self.folder)        
        self.critic_B.save(self.folder)        
        self.soft_actor.save(self.folder + '/models/soft_actor.pth')

    def load(self, folder=None):
        if folder is None:
            folder = self.folder
        try:
            self.critic_A.load(folder)        
            self.critic_B.load(folder)        
            self.soft_actor.load(folder + '/models/soft_actor.pth', self.device)
        except FileNotFoundError:
            raise Exception("No model has been saved !") from None

    def plot_Q(self, pause=False):
        if self.state_size == 1 and self.action_size == 1:
            self.plotter.plot_soft_actor_1D(self.soft_actor, pause)
            self.plotter.plot_Q_1D(self.soft_Q_net1, pause)

        if self.state_size == 2 and self.action_size == 2:
            self.plotter.plot_soft_Q_2D(self.soft_Q_net1, self.soft_actor, pause)
