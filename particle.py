import torch
import numpy as np
import torch.distributions as D
from conjugates import ConjugateBernoulli, ConjugateGaussian, ConjugateCategorical
from utils_categorical import cat_sample, cat2D_sample, cat3D_sample
from crp import CRP



class Particle():
    """
    Main class implementing a single particle in the CCEB model
    """
    def __init__(self, hyp_gamma, hyp_alpha_o=1.0, hyp_alpha_r=1.0, hyp_niw=None, hyp_bb=None):
        
        self.hyp_gamma = hyp_gamma
        self.hyp_alpha_o = hyp_alpha_o
        self.hyp_alpha_r = hyp_alpha_r
        self.hyp_niw = hyp_niw
        self.hyp_bb = hyp_bb

        self.state_space = [0, 1]
        self.state_probs = torch.tensor([0.5, 0.5])
        self.n_states = len(self.state_space)

        self.context_o_space = []
        self.context_r_space = []

        self.context_o_CRP = CRP(hyp_alpha=self.hyp_alpha_o)
        self.context_r_CRP = CRP(hyp_alpha=self.hyp_alpha_r)

        self.prev_o_context = 0
        self.prev_r_context = 0

        self.observ_models : list[list[ConjugateGaussian]] = [[] for _ in self.state_space]
        self.reward_models : list[list[ConjugateBernoulli]] = [[] for _ in self.state_space]

        self.novel_observ_model = ConjugateGaussian(**self.hyp_niw)
        self.novel_reward_model = ConjugateBernoulli(**self.hyp_bb)

    
    @property
    def n_contexts_o(self):
        return len(self.context_o_space)

    @property
    def n_contexts_r(self):
        return len(self.context_r_space)
    



    def create_new_contexts(self, c_o_t, c_r_t):
        """
        Create new observation and reward contexts if needed
        """

        # Handle new observation context if needed
        if c_o_t == self.n_contexts_o: 
            self.context_o_space.append(self.n_contexts_o)
            for i, state in enumerate(self.state_space):
                self.observ_models[i].append(ConjugateGaussian(**self.hyp_niw))
        
        # Handle new reward context if needed
        if c_r_t == len(self.context_r_space): 
            self.context_r_space.append(len(self.context_r_space))
            for i, state in enumerate(self.state_space):
                self.reward_models[i].append(ConjugateBernoulli(**self.hyp_bb))



    def observation_likelihood(self, o_t : torch.Tensor):
        """
        P(o_t | s_t=i, c_o_t=j)

            What is the likelihood of the observation o_t under each state i and observation context j?

        returns a tensor of shape (n_states, n_contexts_o + 1) containing the likelihoods for each state and context (including a new context)
        """
        likelihoods = torch.zeros(self.n_states, (self.n_contexts_o + 1))

        for i in range(self.n_states):
            for j in range(self.n_contexts_o):
                """
                P(o_t | s_t=i, c_o_t=j) = ∫ P(o_t | ω_i,j) P(ω_i,j) dΩ_i,j
                """
                likelihoods[i, j] = self.observ_models[i][j].predictive_likelihood(o_t)
            """
            P(o_t | s_t=i, c_o_t=new) = ∫ P(o_t | ω_i,new) P(ω_i,new) dΩ_i,new
            """
            likelihoods[i, -1] = self.novel_observ_model.predictive_likelihood(o_t)

        return likelihoods

    def reward_likelihood(self, a_t: int, r_t : float):
        """
        P(r_t | s_t=i, c_r_t=k)

            What is the likelihood of the reward r_t under each state i and reward context k?

        returns a tensor of shape (n_states, n_contexts_r + 1) containing the likelihoods for each state and context (including a new context)
        """
        likelihoods = torch.zeros(self.n_states, (self.n_contexts_r + 1))

        for i in range(self.n_states):
            for k in range(self.n_contexts_r):
                """
                P(r_t | s_t=i, c_r_t=k, a_t) = ∫ P(r_t | υ_i,k,a) P(υ_i,k,a) dυ_i,k,a        
                """
                #print("Reward model likelihood for a_t=", a_t, " r_t=", r_t)
                likelihoods[i, k] = self.reward_models[i][k].predictive_likelihood(a_t, r_t)
            """
            P(r_t | s_t=i, c_r_t=new, a_t) = ∫ P(r_t | υ_i,new,a) P(υ_i,new,a) dυ_i,new,a   
            """
            #print("Novel reward model likelihood for a_t=", a_t, " r_t=", r_t)
            likelihoods[i, -1] = self.novel_reward_model.predictive_likelihood(a_t, r_t)

        return likelihoods

    def before_action(self, o_t: torch.Tensor):
        """
        Compute
        P(S_t, C_o_t, C_r_t, J_t | o_{1:t}, a_{1:t-1}, r_{1:t-1})
        """

        # Calculate observation likelihoods for each state and observation context
        """
        P(o_t | S_t=i, C^o_t=j) 
        """
        obs_likelihood = self.observation_likelihood(o_t)  # (n_states, n_contexts_o+1)
        self._obs_likelihood = obs_likelihood  # cache for after_action

        # Compute marginal likelihoods (evidence) for each jump event
        """
        P(o_t | J=0) = Σ_i P(o_t | S_t=i, C_o_t=c_o_{t-1}) P(S_t=i)
        """
        if self.n_contexts_o == 0:
            stay_evidence = torch.tensor(0.0)
        else:
            stay_evidence = torch.dot(self.state_probs, obs_likelihood[:, self.prev_o_context])
        """
        P(o_t | J=1) = Σ_i Σ_j P(o_t | S_t=i, C_o_t=j) P(S_t=i) P(C_o_t=j | θ^o)
        """
        jump_evidence = torch.dot(self.context_o_CRP.probabilities, torch.matmul(self.state_probs, obs_likelihood))

        # Mixture evidence
        mixture_evidence = ((1.0 - self.hyp_gamma)) * stay_evidence + ((self.hyp_gamma) * jump_evidence)
        self.weight = mixture_evidence

        """
        P(J=1 | o_t)
        """
        jump_posterior_o = torch.clamp((self.hyp_gamma * jump_evidence) / (mixture_evidence + 1e-30), 0.0, 1.0)
        self.p_jump_o = jump_posterior_o

        # Compute belief over states conditional on jump event and observation
        """
        P(S_t=i | o_t, J=0, ...) ∝ P(o_t | S_t=i, C_o_t=c_o_{t-1}) P(S_t=i)
        """
        if self.n_contexts_o == 0:
            b0o = torch.zeros_like(self.state_probs)
        else:
            un_b0o = self.state_probs * obs_likelihood[:, self.prev_o_context]
            b0o = un_b0o / (un_b0o.sum() + 1e-30)
        
        self.b0o = b0o

        """
        P(S_t=i | o_t, J=1, ...) ∝ Σ_j P(o_t | S_t=i, C_o_t=j) P(C_o_t=j | θ^o) P(S_t=i)
        """
        un_b1o = torch.matmul(obs_likelihood, self.context_o_CRP.probabilities) * self.state_probs
        b1o = un_b1o / (un_b1o.sum() + 1e-30)

        self.b1o = b1o


    def sample_action(self):
        """
        Thompson sampling using the temporary belief induced by o_t
        """
        # Sample provisional jump
        j_hat = cat_sample(torch.tensor([1.0 - self.p_jump_o, self.p_jump_o]))

        # Sample state conditional on that branch
        b = self.b1o if j_hat == 1 else self.b0o
        s_hat = cat_sample(b)

        # Sample reward context conditional on that branch
        if j_hat == 0:
            c_r_hat = self.prev_r_context
        else:
            c_r_hat = self.context_r_CRP.sample()

        # Thompson sample reward parameters for (s_hat, c_r_hat)
        if c_r_hat == self.n_contexts_r:
            predicted_rewards = self.novel_reward_model.sample_posterior_distribution() # special case for novel context
        else:
            predicted_rewards = self.reward_models[s_hat][c_r_hat].sample_posterior_distribution()
        

        # Select best action
        a_t = int(torch.argmax(predicted_rewards).item())

        return a_t


    def after_action(self, o_t: torch.Tensor, a_t: int, r_t: float):
        """
        Compute
        P(S_t, C_o_t, C_r_t, J_t | o_{1:t}, a_{1:t}, r_{1:t})
        """

        """
        P(r_t | S_t, C_r_t, a_t)
        """
        rew_likelihood = self.reward_likelihood(a_t, r_t)  # (n_states, n_contexts_r+1)

        # Sample jump, contexts, and state
        # 1) Sample Jump
        """
        P(J_t=0 | o_t, a_t, r_t) ∝ Σ_i P(r_t | S_t=i, c_r_{t-1}, a_t) P(o_t | S_t=i, c_o_{t-1}) P(S_t=i)
        """
        if self.n_contexts_r == 0:
            stay_evidence = torch.tensor(0.0)
        else:
            stay_evidence = torch.dot(self.b0o, rew_likelihood[:, self.prev_r_context])

        """
        P(J_t=1 | o_t, a_t, r_t) ∝ Σ_k Σ_j P(r_t | S_t=i, C_r_t=k, a_t) P(o_t | S_t=i, C_o_t=j) P(C_o_t=j | θ^o) P(S_t=i) 
        """
        jump_evidence = torch.dot(self.context_r_CRP.probabilities, torch.matmul(self.b1o, rew_likelihood))

        # predictive reward evidence given o: p(r | o) = (1-p_jump_o)Zr0 + p_jump_o Zr1
        mixture_evidence = (1.0 - self.p_jump_o) * stay_evidence + self.p_jump_o * jump_evidence
        self.weight *= mixture_evidence

        # posterior jump probability given o and r
        p_jump_or = torch.clamp((self.p_jump_o * jump_evidence) / (mixture_evidence + 1e-30), 0.0, 1.0)

        # sample final J_t for this particle
        j_t = cat_sample(torch.tensor([1.0 - p_jump_or, p_jump_or]))
        
        # 2) Sample contexts given jump
        if j_t == 0:
            """
            P(C_o_t = j, C_r_t = k | c_o_{t-1}, c_r_{t-1}, J_t=0, ...) = δ (C_o_t, c_o_{t-1}) δ (C_r_t, c_r_{t-1})
            """
            c_o_t = self.prev_o_context
            c_r_t = self.prev_r_context
        else:
            """
            P(C_o_t = j, C_r_t = k | c_o_{t-1}, c_r_{t-1}, J_t=1, ...) 
                ∝ P(C_o_t=j | θ^o) P(C_r_t=k | θ^r) Σ_i P(S_t=i) P(o_t | S_t=i, C_o_t=j) P(r_t | S_t=i, C_r_t=k, a_t)
            """
            context_probs = \
                self.context_o_CRP.probabilities[:, None] * \
                self.context_r_CRP.probabilities[None, :] * \
                torch.einsum("i,ij,ik->jk", self.state_probs, self._obs_likelihood, rew_likelihood) 
            
            c_o_t, c_r_t = cat2D_sample(context_probs)
        
        # 3) sample state given committed contexts:
        """
        P(S_t=i | o_t, r_t, a_t, c_o_t, c_r_t) ∝ P(S_t=i) P(o_t | S_t=i, c_o_t=j) P(r_t | S_t=i, c_r_t=k, a_t)
        """
        state_prob = self.state_probs * self._obs_likelihood[:, c_o_t] * rew_likelihood[:, c_r_t]
        s_t = cat_sample(state_prob)


        # Update parameters     
        # 1) Update CRPs if needed
        if j_t == 1:
            self.context_o_CRP.update_single(c_o_t)
            self.context_r_CRP.update_single(c_r_t)
            # create new contexts in the models if needed
            self.create_new_contexts(c_o_t, c_r_t)
        # 2) Update observation model
        self.observ_models[s_t][c_o_t].update(o_t)
        # 3) Update reward model
        self.reward_models[s_t][c_r_t].update(a_t, r_t)

        # carry forward contexts
        self.prev_o_context = c_o_t
        self.prev_r_context = c_r_t

        return s_t, c_o_t, c_r_t, j_t

    


def print_particle_params(particle: Particle):
    """
    Print the parameters of the particle's models in a readable format
    """
    print("> Observation Models:")
    for i, state in enumerate(particle.state_space):
        for j, context in enumerate(particle.context_o_space):
            # Extract and print format the observation model parameters (df, loc)
            observation_model_params = particle.observ_models[i][j]._predictive_distribution_params()
            df = observation_model_params['df']
            loc = observation_model_params['loc'].detach().numpy()
            loc = np.array2string(loc, formatter={'float_kind':lambda x: f"{x: .2f}"})
            
            print(f"State {state}, Observ Context {context}, Observation Model Params: df: {df}, loc: {loc}")

    # Extract and print the CRP probabilities and sufficient statistics for observation contexts
    o_prob = particle.context_o_CRP.probabilities.detach().numpy()
    o_prob = np.array2string(o_prob, formatter={'float_kind':lambda x: f"{x: .2f}"})
    o_ss = particle.context_o_CRP.counts.detach().numpy()

    print("\nObser. Context CRP Probabilities:", o_prob, "Suff Stats:", o_ss)

    print("\n> Reward Models:")
    for i, state in enumerate(particle.state_space):
        for j, context in enumerate(particle.context_r_space):
            # Extract and print format the reward model mean and sufficient statistics
            reward_model_mean = particle.reward_models[i][j]._predictive_distribution_params().detach().numpy()
            reward_model_mean = np.array2string(reward_model_mean, formatter={'float_kind':lambda x: f"{x: .2f}"})
            suff_stats_a = particle.reward_models[i][j].posterior_params()['alpha_n'].detach().numpy()
            suff_stats_b = particle.reward_models[i][j].posterior_params()['beta_n'].detach().numpy()
            suff_stats_a = np.round(suff_stats_a, 2)
            suff_stats_b = np.round(suff_stats_b, 2)

            print(f"State {state}, Reward Context {context}, Reward Model Mean: {reward_model_mean}, Suff Stats: {suff_stats_a}, {suff_stats_b}")

    # Extract and print the CRP probabilities and sufficient statistics for reward contexts
    r_prob = particle.context_r_CRP.probabilities.detach().numpy()
    r_prob = np.array2string(r_prob, formatter={'float_kind':lambda x: f"{x: .2f}"})
    r_ss = particle.context_r_CRP.counts.detach().numpy()

    print("\nReward Context CRP Probabilities:", r_prob, "Suff Stats:", r_ss)



