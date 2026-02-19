from copy import deepcopy

import torch
import torch.distributions as D

from particle import Particle
from utils_categorical import cat_sample

class Ensemble():
    """
    Ensemble of particles for particle filtering
    """
    def __init__(self, N, hyp_gamma=0.05, hyp_alpha_o=1.0, hyp_alpha_r=1.0, hyp_niw=None, hyp_bb=None):
        self.N = N
        self.particles = [Particle(hyp_gamma, hyp_alpha_o, hyp_alpha_r, hyp_niw, hyp_bb) for _ in range(N)]
        self.log_weights = torch.zeros(N)

        self.vec_s_t    = torch.zeros(N, dtype=torch.long)
        self.vec_c_o_t  = torch.zeros(N, dtype=torch.long)
        self.vec_c_r_t  = torch.zeros(N, dtype=torch.long)
        self.vec_j_t    = torch.zeros(N, dtype=torch.long)

    @property
    def weights(self):
        """
        Normalised weights for each particle
        """
        return torch.exp(self.log_weights - torch.logsumexp(self.log_weights, dim=0))

    def resample_particles(self):
        """
        Resample particles according to their weights
        """
        
        # Resample particles with replacement according to weights
        indices = D.Categorical(logits=self.log_weights).sample((self.N,))
        self.particles = [deepcopy(self.particles[i]) for i in indices]
        
        # Reset log_weights to uniform after resampling
        self.log_weights = torch.zeros(self.N)

    def before_action(self, o_t : torch.Tensor):
        """
        Before an action is taken, process the new observation o_t
        """
        # Process the new observation for each particle
        for particle in self.particles:
            particle.before_action(o_t)
            
           
    def select_action(self):
        """
        Select an action
        """

        # Sample an action from each particle
        actions = torch.zeros(self.N, dtype=torch.int32)
        for i, particle in enumerate(self.particles):
            actions[i] = particle.sample_action()

        # Sample a particle to select the action from
        selected_particle_i = cat_sample(self.weights)
        selected_action = actions[selected_particle_i]

        return selected_action, actions

    def after_action(self, a_t, r_t, o_t):
        """
        Update the model after an action a_t is taken and reward r_t is received
        """
        for i, particle in enumerate(self.particles):
            # Process the action and reward for this particle
            s_t, c_o_t, c_r_t, j_t = particle.after_action(o_t, a_t, r_t)
            
            # Bookkeep latent states in vectors
            self.vec_s_t[i]     = s_t
            self.vec_c_o_t[i]   = c_o_t
            self.vec_c_r_t[i]   = c_r_t
            self.vec_j_t[i]     = j_t
            
            self.log_weights[i] = particle.log_weight

        # Resample particles based on updated weights
        self.resample_particles()