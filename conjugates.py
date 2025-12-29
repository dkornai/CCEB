from matplotlib import scale
import torch
import torch.distributions as D
import pyro.distributions as pyroD

class SuffStatsGaussian:
    """
    Class that maintains the sufficient statistics for a (multivariate) Gaussian distribution.
    Sufficient statistics are:
        n: number of observations
        sum_x: sum of observations
        sum_xx: sum of outer products of observations
    """
    def __init__(self, d: int):
        self.n = 0
        self.sum_x = torch.zeros(d)
        self.sum_xx = torch.zeros(d, d)

    def update(self, x: torch.Tensor):
        """
        Update sufficient statistics with a new observation x.
        x: tensor of shape [d]
        """
        self.n += 1
        self.sum_x += x
        self.sum_xx += torch.outer(x, x)

    @property
    def mean(self):
        """
        Mean of the observations.
        """
        if self.n == 0:
            return None
        return self.sum_x / self.n

    @property
    def scatter(self):
        """
        S = Σ_i (x_i - x̄)(x_i - x̄)^T = Σ_i (x_i)(x_i)^T - n * (x̄)(x̄)^T

            Scatter around the empirical mean.
        """
        if self.n == 0:
            return None
        mu = self.mean
        return self.sum_xx - self.n * torch.outer(mu, mu)


class ConjugateGaussian:
    def __init__(self, mu0, kappa0, nu0, Lambda0):
        """
        Conjugate Gaussian-Inverse Wishart prior for multivariate Gaussian with unknown mean & covariance.

        mu0:        prior mean mean [d]
        kappa0:     pseudocount of prior measurements
        nu0:        degrees of freedom > d-1
        Lambda0:    scale matrix [d,d], positive definite
        """
        assert type(mu0) == torch.Tensor, "prior mean mu0 must be a torch tensor"
        assert mu0.dim() == 1, "prior mean mu0 must be a 1D tensor"
        self.mu0 = mu0
        
        assert type(kappa0) == float, "pseudocount of prior measurements kappa0 must be a float"
        assert kappa0 > 0, "pseudocount of prior measurements kappa0 must be > 0"
        self.kappa0 = float(kappa0)
        
        assert type(nu0) == float, "nu0 must be a float"
        assert nu0 > mu0.shape[0] - 1, f"degrees of freedom nu0 must be > {mu0.shape[0]-1}"
        self.nu0 = float(nu0)
        
        assert type(Lambda0) == torch.Tensor, "scale matrix Lambda0 must be a torch tensor"
        assert Lambda0.dim() == 2, "scale matrix Lambda0 must be a 2D tensor"
        assert Lambda0.shape[0] == Lambda0.shape[1] == mu0.shape[0], "scale matrix Lambda0 must be of shape (d,d)"
        # check positive definiteness
        eigvals = torch.linalg.eigvalsh(Lambda0)
        assert torch.all(eigvals > 0), "scale matrix Lambda0 must be positive definite"
        self.Lambda0 = Lambda0

        self.d = mu0.shape[0]

        # Maintain sufficient statistics
        self.suffstats = SuffStatsGaussian(self.d)

        # Posterior params start at prior
        self.reset_posterior()

    def reset_posterior(self):
        self.mu_n = self.mu0.clone()
        self.kappa_n = self.kappa0
        self.nu_n = self.nu0
        self.Lambda_n = self.Lambda0.clone()

    def update(self, x: torch.Tensor):
        """
        Add a new observation and update posterior parameters.
        """
        self.suffstats.update(x)
        self._update_posterior()

    def _update_posterior(self):
        """
        Update posterior parameters based on current sufficient statistics.
        """
        n = self.suffstats.n
        x_bar = self.suffstats.mean
        

        # Update posterior of mean
        """
        μ_n = ( (κ_0 / (κ_0 + n)) * μ0 ) + ( (n / (κ0 + n)) * x̄ )
        """
        self.mu_n = ((self.kappa0 / (self.kappa0 + n)) * self.mu0 ) + ((n / (self.kappa0 + n)) * x_bar)

        # Update posterior of kappa
        """
        κ_n = κ_0 + n
        """
        self.kappa_n = self.kappa0 + n

        # Update posterior of nu
        """
        ν_n = ν_0 + n
        """
        self.nu_n = self.nu0 + n

        # Update posterior of Lambda
        """
        Λ_n = Λ_0 + S + ((κ_0 * n )/ (κ_n)) * (x̄ - μ0)(x̄ - μ0)^T

        """
        S = self.suffstats.scatter
        diff = (x_bar - self.mu0).unsqueeze(1)  # column vector, difference between sample mean and prior mean
        self.Lambda_n = self.Lambda0 + S + ((self.kappa0 * n )/ (self.kappa_n)) * (diff @ diff.T)

    def posterior_params(self):
        return {
            "mu_n": self.mu_n,
            "kappa_n": self.kappa_n,
            "nu_n": self.nu_n,
            "Lambda_n": self.Lambda_n,
        }
    
    def sample_posterior_distribution(self):
        """
        Sample from the posterior distribution over the mean vector and covariance matrix.
        """
        # Sample covariance matrix
        """
        To get: Σ ~ IW(ν_n, Λ_n)    
        Ω ~ W(ν_n, Λ_n^{-1})        we can sample a precision matrix from the Wishart distribution,
        Σ = Ω^{-1}                  then invert
        """
        Prec = D.Wishart(df=self.nu_n, covariance_matrix=torch.linalg.inv(self.Lambda_n)).sample()
        Sigma = torch.linalg.inv(Prec)
        # Sample mean from Gaussian
        """
        μ ~ 𝒩(μ_n, Σ / κ_n)
        """
        mu = D.MultivariateNormal(loc=self.mu_n, covariance_matrix=(Sigma / self.kappa_n)).sample()
        
        return mu, Sigma
    
    def _predictive_distribution_params(self):
        """
        Returns parameters of the multivariate Student-t predictive distribution (uncerteanty over mean and variance is integrated out).
        """
        df = self.nu_n - self.d + 1
        scale = (self.Lambda_n * (self.kappa_n + 1)) / (self.kappa_n * df)
        
        return {
            "df": df,
            "loc": self.mu_n,
            "scale": scale,
        }
    
    def predictive_likelihood(self, x: torch.Tensor):
        """
        Returns the predictive likelihood of a new observation x.
        """
        params = self._predictive_distribution_params()
        scale_tril = torch.linalg.cholesky(params["scale"])
        pred_dist = pyroD.MultivariateStudentT(df=params["df"], loc=params["loc"], scale_tril=scale_tril)
        return torch.exp(pred_dist.log_prob(x))
    


class SuffStatsCategorical:
    """
    Maintains sufficient statistics for a categorical distribution.
    Sufficient stats: counts of each category.
    """
    def __init__(self, K: int):
        self.K = K
        self.counts = torch.zeros(K)

    def update(self, x: int, confidence: float = 1.0):
        """
        Update with a new categorical observation (int in [0, K-1]).
        """
        assert 0 <= x < self.K, f"observation {x} out of range for {self.K} categories"
        self.counts[x] += confidence

    @property
    def n(self):
        return int(self.counts.sum())

    def as_tensor(self):
        return self.counts.clone()


class ConjugateCategorical:
    """
    Conjugate model: categorical likelihood with Dirichlet prior.
    Supports incremental updates and posterior predictive probabilities.
    """
    def __init__(self, alpha0: torch.Tensor):
        assert alpha0.dim() == 1, "alpha0 must be a 1D tensor"
        assert torch.all(alpha0 > 0), "alpha0 parameters must be > 0"

        self.alpha0 = alpha0.clone().to(torch.float)
        self.K = alpha0.shape[0]

        self.suffstats = SuffStatsCategorical(self.K)

        # Start at prior
        self.reset_posterior()

    def reset_posterior(self):
        self.alpha_n = self.alpha0.clone()

    def update(self, x: int, confidence: float = 1.0):
        """
        Add a new observation and update posterior parameters.
        """
        self.suffstats.update(x, confidence)
        self._update_posterior()

    def _update_posterior(self):
        self.alpha_n = self.alpha0 + self.suffstats.as_tensor().to(torch.float)

    def posterior_params(self):
        return {
            "alpha_n": self.alpha_n
        }

    def sample_posterior_distribution(self):
        """
        Sample from the posterior distribution over theta.
        """
        theta = D.Dirichlet(self.alpha_n).sample()
        return theta

    def _predictive_distribution_params(self):
        """
        Posterior predictive probabilities for each category:
        theta_n = (alpha_n / sum(alpha_n))
        """
        theta_n = self.alpha_n / self.alpha_n.sum()
        return theta_n

    def predictive_likelihood(self, x: int):
        """
        Predictive likelihood of a new categorical observation.
        """
        probs = self._predictive_distribution_params()
        return probs[x].item()
