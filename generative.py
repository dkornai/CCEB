import torch
import numpy as np
import torch.distributions as D

from crp import CRP
from conjugates import ConjugateGaussian, ConjugateCategorical
import particle
from utils_categorical import cat_sample


class GenerativeModel():
    """
    Main class implementing the generative model inverted by the particle filter.
    """
    def __init__(self, hyp_gamma, hyp_alpha_o, hyp_alpha_r, hyp_obs, hyp_rew):
        
        self.hyp_gamma = hyp_gamma
        self.hyp_alpha_o = hyp_alpha_o
        self.hyp_alpha_r = hyp_alpha_r
        self.hyp_obs = hyp_obs
        self.hyp_rew = hyp_rew

        self.state_space = [0, 1]
        self.state_probs = torch.tensor([0.5, 0.5])
        self.n_states = len(self.state_space)

        self.context_o_space = []
        self.context_r_space = []

        self.observation_models = [[] for _ in self.state_space]
        self.reward_models      = [[] for _ in self.state_space]

        self.context_o_CRP = CRP(hyp_alpha=self.hyp_alpha_o)
        self.context_r_CRP = CRP(hyp_alpha=self.hyp_alpha_r)

        self.obs_model_prior    = ConjugateGaussian(**self.hyp_obs)
        self.reward_model_prior = ConjugateCategorical(alpha0=self.hyp_rew)

    @property
    def n_contexts_o(self):
        return len(self.context_o_space)

    @property
    def n_contexts_r(self):
        return len(self.context_r_space)

    def sample_jump_times(self, T:int) -> np.ndarray:
        """
        Sample jump times for a sequence of length T
        """
        jump_times = []
        for t in range(T):
            if t == 0:
                j_t = 1
            else:
                j_t = cat_sample(torch.tensor([1 - self.hyp_gamma, self.hyp_gamma]))
            jump_times.append(j_t)
        jump_times = np.array(jump_times)
        
        return jump_times

    def sample_obs_contexts(self, jump_times:np.ndarray) -> np.ndarray:
        """
        Sample observation contexts for a sequence given jump times
        """
        CRP_o = CRP(hyp_alpha=self.hyp_alpha_o)
        
        obs_contexts = []
        for t, j_t in enumerate(jump_times):
            if j_t == 0:
                c_o_t = obs_contexts[-1]
            
            else:
                c_o_t = CRP_o.sample()
                # Update CRP probabilities
                CRP_o.update_single(c_o_t)

            obs_contexts.append(c_o_t)

        obs_contexts = np.array(obs_contexts)

        return obs_contexts

    def sample_rew_contexts(self, jump_times:np.ndarray) -> np.ndarray:
        """
        Sample reward contexts for a sequence given jump times
        """
        CRP_r = CRP(hyp_alpha=self.hyp_alpha_r)

        rew_contexts = []
        for t, j_t in enumerate(jump_times):
            if j_t == 0:
                c_r_t = rew_contexts[-1]
            else:
                c_r_t = CRP_r.sample()
                # Update CRP probabilities
                CRP_r.update_single(c_r_t)

            rew_contexts.append(c_r_t)

        rew_contexts = np.array(rew_contexts)

        return rew_contexts

    def sample_state(self, T:int) -> np.ndarray:
        """
        Sample a sequence of latent states of length T
        """
        states = []
        for t in range(T):
            s_t = cat_sample(self.state_probs)
            states.append(s_t)
        states = np.array(states)

        return states
    
    def sample_observation_models(self, context_o_list:np.ndarray) -> list[list[D.MultivariateNormal]]:
        """
        Sample observation models for each state in each observation context
        """
        # max context index
        n_contexts_o = np.max(context_o_list) + 1

        observation_models = [[] for _ in self.state_space]
        for state in self.state_space:
            for c_o in range(n_contexts_o):
                if c_o >= len(observation_models[state]):
                    # Sample new model from prior
                    mu, Sigma = self.obs_model_prior.sample_posterior_distribution()
                    obs_model = D.MultivariateNormal(loc=mu, covariance_matrix=Sigma)
                    observation_models[state].append(obs_model)

        return observation_models
    
    def sample_reward_models(self, context_r_list:np.ndarray) -> list[list[np.ndarray]]:
        """
        Sample reward models for each state in each reward context
        """
        # max context index
        n_contexts_r = np.max(context_r_list) + 1

        reward_models = [[] for _ in self.state_space]
        for state in self.state_space:
            for c_r in range(n_contexts_r):
                if c_r >= len(reward_models[state]):
                    # Sample new model from prior
                    mean_rew = self.reward_model_prior.sample_posterior_distribution().detach().numpy()
                    reward_models[state].append(mean_rew)

        return reward_models
    
    def sample_observations(self, states:np.ndarray, context_o_list:np.ndarray, obs_models:list[list[D.MultivariateNormal]]) -> np.ndarray:
        """
        Sample observations given states and observation contexts
        """
        assert len(states) == len(context_o_list), "Length of states and context_o_list must be equal"
        assert len(obs_models) == self.n_states, "Number of observation model lists must equal number of states"
        assert len(obs_models[0]) == np.max(context_o_list) + 1, "Not enough observation models for the given contexts"

        observations = []
        for t in range(len(states)):
            s_t = states[t]
            c_o_t = context_o_list[t]
            obs_model = obs_models[s_t][c_o_t]
            o_t = obs_model.sample().numpy()
            observations.append(o_t)
        observations = np.array(observations)

        return observations

    def sample_rewards(self, states:np.ndarray, context_r_list:np.ndarray, rew_models:list[list[np.ndarray]], actions:np.ndarray) -> np.ndarray:
        """
        Sample rewards given states, reward contexts, and actions
        """
        assert len(states) == len(context_r_list) == len(actions), "Length of states, context_r_list, and actions must be equal"
        assert len(rew_models) == self.n_states, "Number of reward model lists must equal number of states"
        assert len(rew_models[0]) == np.max(context_r_list) + 1, "Not enough reward models for the given contexts"

        rewards = []
        for t in range(len(states)):
            s_t = states[t]
            c_r_t = context_r_list[t]
            a_t = actions[t]
            rew_model = rew_models[s_t][c_r_t]
            # Here we could condition on action if needed; for now we ignore it
            r_t = D.Bernoulli(probs=torch.tensor(rew_model[a_t])).sample().item()
            rewards.append(r_t)
        rewards = np.array(rewards)

        return rewards