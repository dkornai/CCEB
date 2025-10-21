import torch
import torch.distributions as D

class Categorical2D():
    """
    Class for representing a 2D categorical distribution [e.g. a joint P(X,Y)].
    
    probs: 2D tensor of probabilities (not necessarily normalized)
    """
    def __init__(self, probs: torch.Tensor):
        assert probs.dim() == 2, "probs must be a 2D tensor"
        self.dim1, self.dim2 = probs.shape
        self.probs2d = probs / torch.sum(probs)
        self.probs = probs.flatten()

    def sample(self):
        idx = D.Categorical(probs=self.probs).sample().item()
        i = idx // self.dim2
        j = idx % self.dim2

        return int(i), int(j)

def cat_sample(probs: torch.Tensor):
    """
    Sample from a categorical distribution defined by the given probabilities.
    
    probs: 1D tensor of probabilities
    returns: sampled index
    """
    normalized_probs = probs / torch.sum(probs)
    return int(D.Categorical(probs=normalized_probs).sample().item())

def cat2D_sample(probs: torch.Tensor):
    """
    Sample from a 2D categorical distribution defined by the given probabilities.
    
    probs: 2D tensor of probabilities
    returns: sampled indices (i, j)
    """
    cat2d = Categorical2D(probs)
    return cat2d.sample()