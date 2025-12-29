from copy import deepcopy

import torch
import torch.distributions as D

from particle import Particle

class Ensemble():
    """
    Ensemble of particles for particle filtering
    """
    def __init__(self, N, hyp_gamma=0.05, hyp_alpha_o=1.0, hyp_alpha_r=1.0, hyp_niw=None, hyp_dm=None):
        self.N = N
        self.particles = [Particle(hyp_gamma, hyp_alpha_o, hyp_alpha_r, hyp_niw, hyp_dm) for _ in range(N)]
        self.weights = torch.ones(N) / N

        self.vec_s_t    = torch.zeros(N, dtype=torch.long)
        self.vec_c_o_t  = torch.zeros(N, dtype=torch.long)
        self.vec_c_r_t  = torch.zeros(N, dtype=torch.long)
        self.vec_j_t    = torch.zeros(N, dtype=torch.long)

    def resample_particles(self):
        """
        Resample particles according to their weights
        """
        # Normalise weights
        self.weights /= torch.sum(self.weights)
        
        # Resample particles with replacement according to weights
        indices = D.Categorical(probs=self.weights).sample((self.N,))
        self.particles = [deepcopy(self.particles[i]) for i in indices]
        
        # Reset weights to uniform after resampling
        self.weights = torch.ones(self.N) / self.N

    def before_action(self, o_t : torch.Tensor):
        """
        Before an action is taken, process the new observation o_t
        """

        # 1) Resample particles based on predictive evidence (auxiliary particle filter)
        for i, particle in enumerate(self.particles):
            self.weights[i] = particle.predictive_evidence(o_t)

        self.resample_particles()

        # 2) Sample latent variables and update observation models
        for i, particle in enumerate(self.particles):
            # Sample latent states
            s_t, c_o_t, c_r_t, j_t = particle.sample_latent_states(o_t)

            # Update observation model parameters
            particle.update_o_params(s_t, c_o_t, o_t, j_t)

            # Bookkeep latent states in vectors
            self.vec_s_t[i]     = s_t
            self.vec_c_o_t[i]   = c_o_t
            self.vec_j_t[i]     = j_t
            self.vec_c_r_t[i]   = c_r_t 
           
    def select_action(self):
        """
        Select an action
        """

        # Sample an action from each particle
        actions = torch.zeros(self.N, dtype=torch.long)
        for i, particle in enumerate(self.particles):
            actions[i] = particle.sample_action(self.vec_s_t[i], self.vec_c_r_t[i])

        # Sample a particle to select the action from
        selected_action = actions[D.Categorical(probs=self.weights).sample().item()]

        return selected_action, actions

    def after_action(self, a_t, r_t):
        """
        Update the model after an action a_t is taken and reward r_t is received
        """
        for i, particle in enumerate(self.particles):
            # 1) Calculate weight based on reward likelihood
            self.weights[i] = particle.reward_likelihood(r_t, self.vec_s_t[i], a_t, self.vec_c_r_t[i])
            
            # 2) Update reward model parameters
            particle.update_r_params(self.vec_c_r_t[i], self.vec_s_t[i], a_t, r_t, self.vec_j_t[i])

            # 3) Update previous contexts
            particle.prev_o_context = self.vec_c_o_t[i]
            particle.prev_r_context = self.vec_c_r_t[i]

        # Resample particles based on updated weights
        self.resample_particles()
