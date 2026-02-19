import torch
import torch.distributions as D

class CRP():
    def __init__(self, hyp_alpha):
        self.hyp_alpha = hyp_alpha
        self.n = 0
        self.counts = torch.zeros(1)

    @property
    def probs(self):
        """
        Prior probabilities for each category, including a new one
        """
        # returns the probability of each context (including a new one)
        prob = torch.zeros(self.n+1)
        prob[:-1] = self.counts
        prob[-1] = self.hyp_alpha
        prob = prob / prob.sum()
        
        return prob
    
    def update_vec(self, c:torch.Tensor):
        """
        Update the counts with a vector of context responsibilities (either the same length as n, or n+1)
        """
        # if the length of the context responsibilities vector is == n, add c to counts
        if c.shape[0] == self.n:
            self.counts += c
        # if a new context is needed (length of c == n+1), expand counts and add c
        else:
            self.n += 1
            new_counts = torch.zeros(self.n)
            new_counts[:-1] = self.counts
            new_counts += c
            self.counts = new_counts

    def update_single(self, c:int):
        """
        Update the counts with a single context assignment
        """
        if c == self.n:
            self.n += 1
            new_counts = torch.zeros(self.n)
            new_counts[:-1] = self.counts
            new_counts[c] += 1
            self.counts = new_counts
        else:
            self.counts[c] += 1

    def sample(self) -> int:
        """
        Sample from the CRP prior
        """
        return D.Categorical(probs=self.probs).sample().item()
