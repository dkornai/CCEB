import torch
import torch.distributions as D


def cat_sample(probs: torch.Tensor):
    """
    Sample from a categorical distribution defined by the given probabilities.
    
    probs: 1D tensor of probabilities
    returns: sampled index
    """
    normalized_probs = probs / torch.sum(probs)
    return int(D.Categorical(probs=normalized_probs).sample().item())


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

def cat2D_sample(probs: torch.Tensor):
    """
    Sample from a 2D categorical distribution defined by the given probabilities.
    
    probs: 2D tensor of probabilities
    returns: sampled indices (i, j)
    """
    cat2d = Categorical2D(probs)
    return cat2d.sample()

class Categorical3D():
    """
    Class for representing a 3D categorical distribution [e.g. a joint P(X,Y,Z)].
    
    probs: 3D tensor of probabilities (not necessarily normalized)
    """
    def __init__(self, probs: torch.Tensor):
        assert probs.dim() == 3, "probs must be a 3D tensor"
        self.dim1, self.dim2, self.dim3 = probs.shape
        self.probs3d = probs / torch.sum(probs)
        self.probs = probs.flatten()

    def sample(self):
        idx = D.Categorical(probs=self.probs).sample().item()
        i = idx // (self.dim2 * self.dim3)
        j = (idx % (self.dim2 * self.dim3)) // self.dim3
        k = idx % self.dim3

        return int(i), int(j), int(k)
    
def cat3D_sample(probs: torch.Tensor):
    """
    Sample from a 3D categorical distribution defined by the given probabilities.
    
    probs: 3D tensor of probabilities
    returns: sampled indices (i, j, k)
    """
    cat3d = Categorical3D(probs)
    return cat3d.sample()