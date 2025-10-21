import torch
import numpy as np
import torch.distributions as D
from conjugates import ConjugateGaussian, ConjugateCategorical
from utils_categorical import cat_sample, cat2D_sample
from crp import CRP



class Particle():
    """
    Main class implementing a single particle in the CCEB model
    """
    def __init__(self, hyp_gamma, hyp_alpha_o=1.0, hyp_alpha_r=1.0):
        
        self.hyp_gamma = hyp_gamma
        self.hyp_alpha_o = hyp_alpha_o
        self.hyp_alpha_r = hyp_alpha_r

        self.state_space = [0, 1]
        self.state_probs = torch.tensor([0.5, 0.5])
        self.n_states = len(self.state_space)

        self.context_o_space = []
        self.context_r_space = []

        self.observation_models = [[] for _ in self.state_space]
        self.reward_models = [[] for _ in self.state_space]

        self.context_o_CRP = CRP(hyp_alpha=self.hyp_alpha_o)
        self.context_r_CRP = CRP(hyp_alpha=self.hyp_alpha_r)

        self.prev_o_context = 0
        self.prev_r_context = 0

        self.novel_obs_model = ConjugateGaussian(**conjugate_gaussian_standard)
        self.novel_reward_model = ConjugateCategorical(alpha0=alpha_0_standard)

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
                self.observation_models[i].append(ConjugateGaussian(**conjugate_gaussian_standard))
        
        # Handle new reward context if needed
        if c_r_t == len(self.context_r_space): 
            self.context_r_space.append(len(self.context_r_space))
            for i, state in enumerate(self.state_space):
                self.reward_models[i].append(ConjugateCategorical(alpha0=alpha_0_standard))



    def observation_likelihood(self, o_t : torch.Tensor):
        """
        P(o_t | s_t=i, c_o_t=j)

            What is the likelihood of the observation o_t under each state i and observation context j?
        """
        likelihoods = torch.zeros(self.n_states, (self.n_contexts_o + 1))

        for i in range(self.n_states):
            for j in range(self.n_contexts_o):
                """
                P(o_t | s_t=i, c_o_t=j) = ∫ P(o_t | Ω_i,j) P(Ω_i,j| ... ) dΩ_i,j            Multivariate Student-t
                """
                likelihoods[i, j] = self.observation_models[i][j].predictive_likelihood(o_t)
            """
            P(o_t | s_t=i, c_o_t=new) = ∫ P(o_t | Ω_i,new) P(Ω_i,new| ... ) dΩ_i,new        Multivariate Student-t
            """
            likelihoods[i, -1] = self.novel_obs_model.predictive_likelihood(o_t)

        return likelihoods

    def marginal_evidence(self, likelihoods):
        """
        P(o_t |c_o_{t-1}, j_t=i) = Σ_s_t Σ_c_o_t P(o_t | S_t, C_o_t) P(S_t) P(C_o_t |c_o_{t-1}, j_t=i)

            What is the marginal likelihood of the observation o_t under each value of the jump variable j_t? \\
            This is used to sample the jump variable j_t to make proposals more efficient.
        """
        
        """
        P(o_t |c_o_{t-1}, j_t=0) = Σ_s_t P(o_t | S_t, c_o_{t-1}) P(S_t)
        """
        # Special case for first time step, a jump must be made, since there is no previous context
        if self.n_contexts_o == 0: 
            lj0 = 0.0
        # Otherwise, marginal for not jumping is the likelihood under the previous context, weighted by state probabilities
        else:
            lj0 = torch.zeros(self.n_states)
            for i in range(self.n_states):
                lj0[i] = self.state_probs[i] * likelihoods[i, self.prev_o_context]
            lj0 = torch.sum(lj0)

        """
        P(o_t |c_o_{t-1}, j_t=1) = Σ_s_t Σ_c_o_t P(o_t | S_t, C_o_t) P(S_t) P(C_o_t | θ^o)
        """
        lj1 = torch.zeros(self.n_states)
        for i in range(self.n_states):
            lj1[i] = self.state_probs[i] * torch.sum(likelihoods[i,:]*self.context_o_CRP.probabilities)
        lj1 = torch.sum(lj1)

        return lj0, lj1
    
    def sample_jump(self, likelihoods):
        """
        j_t ~ P(J_t | o_t, ...) ∝ P(o_t | J_t, ...) P(J_t)

            Sample the jump variable j_t. 0 = no jump, 1 = jump
        """

        lj0, lj1 = self.marginal_evidence(likelihoods)

        """
        P(J_t=0 | o_t, ...) ∝ P(o_t | J_t=0, ...) P(J_t=0) = lj0 * (1 - γ)
        P(J_t=1 | o_t, ...) ∝ P(o_t | J_t=1, ...) P(J_t=1) = lj1 * γ
        """
        j_probs = torch.tensor([lj0*(1-self.hyp_gamma), lj1*self.hyp_gamma])

        return cat_sample(j_probs)

    def sample_state_context_o(self, likelihoods, j_t : int):
        """
        s_t, c_o_t ~ P(S_t, C_o_t | o_t, j_t, c_o_{t-1} ...) ∝ P(o_t | S_t, C_o_t) P(S_t) P(C_o_t | j_t, c_o_{t-1})

            Sample the latent state and the observation context given the jump variable
        """
        if j_t == 0:
            # If no jump was chosen
            """
            s_t, c_o_t ~ P(S_t, C_o_t | o_t, j_t=0, ...) ∝ P(o_t | S_t, C_o_t) P(s_t) δ (C_o_t, c_o_{t-1})
            """
            s_t_probs = torch.zeros(self.n_states)
            for i in range(self.n_states):
                """
                P(s_t=i | o_t, j_t=0, c_o_{t-1}) ∝ P(o_t | s_t=i, c_o_{t-1}) P(s_t=i)
                """
                s_t_probs[i] = self.state_probs[i] * likelihoods[i, self.prev_o_context]

            s_t     = cat_sample(s_t_probs)    
            c_o_t   = self.prev_o_context        

        else: 
            # If jump was chosen
            """
            s_t, c_o_t ~ P(S_t, C_o_t | o_t, j_t=1, ...) ∝ P(o_t | S_t, C_o_t) P(S_t) P(C_o_t | θ^o) 
            """
            joint_probs = torch.zeros((self.n_states, self.n_contexts_o + 1))
            for i in range(self.n_states):
                # Existing observation contexts:
                """
                P(s_t=i, c_o_t=j | o_t, j_t=1, ...) ∝ P(o_t | s_t=i, c_o_t=j) P(s_t=i) P(c_o_t=j | θ^o)
                """
                for j in range(self.n_contexts_o):
                    joint_probs[i, j] = self.state_probs[i] * likelihoods[i, j] * self.context_o_CRP.probabilities[j]
                # New observation context:
                """
                P(s_t=i, c_o_t=new | o_t, j_t=1, ...) ∝ P(o_t | s_t=i, c_o_t=new) P(s_t=i) P(c_o_t=new | θ^o)
                """
                joint_probs[i, -1] = self.state_probs[i] * likelihoods[i, -1] * self.context_o_CRP.probabilities[-1]

            s_t, c_o_t = cat2D_sample(joint_probs)

        return s_t, c_o_t
    
    def sample_context_r(self, j_t : int):
        """
        c_r_t ~ P(c_r_t | j_t, ...) = P(c_r_t | θ^r) if j_t=1 else δ(c_r_t, c_r_{t-1})

            Sample the reward context given the jump variable
        """

        if j_t == 0:
            c_r_t = self.prev_r_context

        else: # if j == 1
            c_r_t = self.context_r_CRP.sample()

        return c_r_t

    

    def sample_latent_states(self, o_t : torch.Tensor):
        """
        s_t, c_o_t, c_r_t, j_t ~ P(S_t, C_o_t, C_r_t, J_t | o_t, ...)

            Sample the latent state, observation context, reward context, and jump variable given the new observation o_t.\\
            This is done after an observation is recieved, but before an action is taken. 
        """
        likelihoods = self.observation_likelihood(o_t)

        # Sample jump variable using marginal likelihoods
        """
        P(J_t | o_t, ...) ∝ P(o_t | J_t, ...) P(J_t)
        """
        j_t = self.sample_jump(likelihoods)
        
        # Sample state and observation context given jump variable
        """
        P(S_t, S_o_t | o_t, j_t, ...) ∝ P(o_t | S_t, C_o_t) P(S_t) P(C_o_t | j_t, c_o_{t-1})
        """
        s_t, c_o_t = self.sample_state_context_o(likelihoods, j_t)
        
        # Sample reward context given jump variable
        """
        P(C_r_t | j_t, ...) = P(C_r_t | θ^r) if j_t=1 else δ(c_r_t, c_r_{t-1})
        """
        c_r_t = self.sample_context_r(j_t)
        
        # Handle new reward context if needed
        self.create_new_contexts(c_o_t, c_r_t)

        return s_t, c_o_t, c_r_t, j_t


    def update_o_params(self, s_t : int, c_o_t : int, o_t : float, j_t : int):
        """
        Update the observation models (NIW) and the observation context CRP
        """       
        
        # Update the observation model parameter distribution Ω_i,j based on the new observation o_t (handled internally by the conjugate model class)
        self.observation_models[s_t][c_o_t].update(o_t)
        
        # Update the CRP counts θ^o if a jump was made
        if j_t == 1:
            self.context_o_CRP.update_single(c_o_t)
            

    def sample_action(self, s_t : int, c_r_t : int):
        """
        Thompson sampling action selection based on the reward model for the current state and reward context
        """

        # Sample the posterior distribution over parameters of the reward model
        """
        υ ~ P(ϒ_i,k | ... ) 
        """
        predicted_rewards = self.reward_models[s_t][c_r_t].sample_posterior_distribution()

        # Choose the action with the highest predicted reward
        """
        a_t = argmax_a E[r_t | s_t=i, c_r_t=k, a_t=a, υ_i,k]
        """
        greedy_a_t = int(torch.argmax(predicted_rewards).item())

        # make epsilon-greedy here if desired
        if torch.rand(1).item() < 0.01:
            # choose random action from [0, 1, 2, 3]
            a_t = int(torch.randint(0, 4, (1,)).item())
        else:
            a_t = greedy_a_t

        return a_t


    def reward_likelihood(self, r_t : float, s_t : int, a_t : int, c_r_t : int):
        """
        P(r_t={0,1} | s_t=i, c_r_t=k, a_t=a, ϒ) 

            Compute the predictive likelihood of a reward r_t given the state s_t, reward context c_r_t, action a_t, and the reward model parameter distribution ϒ_i,k
        """
        if r_t == 1.0:
            return self.reward_models[s_t][c_r_t].predictive_likelihood(a_t)
        else:
            return 1.0 - self.reward_models[s_t][c_r_t].predictive_likelihood(a_t)
        

    def update_r_params(self, c_r_t : int, s_t : int, a_t : int, r_t : float, j_t : int):
        """
        P(ϒ | ...) 

            Update the reward models and the reward context CRP
        """

        if r_t == 1.0: # If the action was optimal, update the model with the actual action taken
            a_t = a_t
            
        else: # If the action was not optimal, sample an alternative action from the model excluding the action taken
            probs = self.reward_models[s_t][c_r_t].posterior_params()['alpha_n']
            probs[a_t] = 0.0
            a_t = cat_sample(probs)

        # Update the reward model parameter distribution ϒ_i,j based on the action taken and reward received
        self.reward_models[s_t][c_r_t].update(a_t)

        # Update the CRP counts θ^r if a jump was made        
        if j_t == 1:
            self.context_r_CRP.update_single(c_r_t)

    def predictive_evidence(self, o_t):
        """
        P(o_t | ...) = 

            Compute the marginal likelihood of the observation o_t under the particle's model.\\
            Used to reweight particles after an observation is recieved.
        """
        likelihoods = self.observation_likelihood(o_t)
        lj0, lj1 = self.marginal_evidence(likelihoods)
        p_evidence = (1 - self.hyp_gamma) * lj0 + self.hyp_gamma * lj1
        
        return p_evidence
    


def print_particle_params(particle: Particle):
    """
    Print the parameters of the particle's models in a readable format
    """
    print("> Observation Models:")
    for i, state in enumerate(particle.state_space):
        for j, context in enumerate(particle.context_o_space):
            # Extract and print format the observation model parameters (df, loc)
            observation_model_params = particle.observation_models[i][j]._predictive_distribution_params()
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
            suff_stats = particle.reward_models[i][j].posterior_params()['alpha_n'].detach().numpy()
            suff_stats = np.round(suff_stats, 1)

            print(f"State {state}, Reward Context {context}, Reward Model Mean: {reward_model_mean}, Suff Stats: {suff_stats}")

    # Extract and print the CRP probabilities and sufficient statistics for reward contexts
    r_prob = particle.context_r_CRP.probabilities.detach().numpy()
    r_prob = np.array2string(r_prob, formatter={'float_kind':lambda x: f"{x: .2f}"})
    r_ss = particle.context_r_CRP.counts.detach().numpy()

    print("\nReward Context CRP Probabilities:", r_prob, "Suff Stats:", r_ss)



# Standard prior parameters for conjugate models
conjugate_gaussian_standard = {
    'mu0': torch.tensor([0.0, 0.0, 0.0]),
    'kappa0': 0.1,
    'nu0': 4.0,
    'Lambda0': torch.eye(3) * 1.0 
}
alpha_0_standard = torch.tensor([0.1, 0.1, 0.1, 0.1])