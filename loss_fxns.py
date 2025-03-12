import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import LRScheduler
from torch.special import gammaln


class LogitLaplaceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(y, x):
        # Clamp x to avoid extreme values near 0 and 1
        x = torch.clamp(x, min=1e-5, max=1-1e-5)

        # Extract mu and b, and clamp the input to exp to prevent it from growing too large
        mu = y[:, 0, :].unsqueeze(1)
        b = torch.clamp(torch.exp(torch.clamp(y[:, 1, :], max=20)), min=1e-4).unsqueeze(1)

        # Compute density
        logit_x = torch.logit(x)  # Clamp prevents instability
        density = (1 / (2 * b * x * (1 - x))) * torch.exp(-torch.abs(logit_x - mu) / b)

        # Clamp density to ensure it's positive and greater than a small value
        density = torch.clamp(density, min=1e-10)

        # Compute loss
        loss = -torch.mean(torch.log(density + 1e-5))

        return loss

class LogitCauchyLoss(nn.Module):
    """
    Single-component Cauchy on the logit scale.
    Model output: (B,2,T) => [mu, log_gamma].
    x in [0,1]: (B,T) or (B,1,T).

    PDF(logit_x) = 1 / [pi * gamma * (1 + ((logit_x - mu)/gamma)^2)]
    plus Jacobian: -log[x(1-x)].
    """
    def __init__(self, eps=1e-6, clamp_mu=(-10,10), clamp_log_gamma=(-10,10),
                 reduction='mean'):
        super().__init__()
        self.eps = eps
        self.clamp_mu = clamp_mu
        self.clamp_log_gamma = clamp_log_gamma
        self.reduction = reduction
        self.LOG_PI = math.log(math.pi)

    def forward(self, y, x):
        B, C, T = y.shape
        assert C == 2, f"Expected 2 channels, got {C}"

        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)  # => (B,T)

        x = torch.clamp(x, self.eps, 1.0 - self.eps)
        logit_x = torch.logit(x)

        mu        = y[:, 0, :]
        log_gamma = y[:, 1, :]

        mu        = torch.clamp(mu, *self.clamp_mu)
        log_gamma = torch.clamp(log_gamma, *self.clamp_log_gamma)

        gamma = torch.exp(log_gamma) + self.eps

        z = (logit_x - mu) / gamma
        # log Cauchy = - log(pi) - log(gamma) - log(1+z^2)
        log_cauchy = -self.LOG_PI - torch.log(gamma) - torch.log1p(z**2)

        jacobian = -torch.log(x*(1 - x) + self.eps)

        log_prob = log_cauchy + jacobian
        nll = -log_prob

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll

class LogitGaussianLoss(nn.Module):
    """
    Single-component Gaussian on the logit scale.

    Model output: (B, 2, T) => [mu, log_sigma].
    x in [0,1]: (B, T) or (B,1,T).

    PDF(logit_x) = Normal( (logit_x - mu)/sigma ),
    plus Jacobian: -log[x(1-x)].
    """
    def __init__(self, eps=1e-6, clamp_mu=(-10,10), clamp_log_sigma=(-10,10),
                 reduction='mean'):
        super().__init__()
        self.eps = eps
        self.clamp_mu = clamp_mu
        self.clamp_log_sigma = clamp_log_sigma
        self.reduction = reduction
        self.LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)

    def forward(self, y, x):
        # y: (B,2,T) => [mu, log_sigma]
        # x: (B,T) or (B,1,T) in [0,1]
        B, C, T = y.shape
        assert C == 2, f"Expected 2 channels, got {C}"

        # If x is (B,1,T), optionally squeeze:
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)  # => (B,T)

        # 1) clamp x & logit
        x = torch.clamp(x, self.eps, 1.0 - self.eps)
        logit_x = torch.logit(x)  # (B,T)

        mu        = y[:, 0, :]
        log_sigma = y[:, 1, :]

        # 2) clamp
        mu = torch.clamp(mu, self.clamp_mu[0], self.clamp_mu[1])
        log_sigma = torch.clamp(log_sigma, self.clamp_log_sigma[0], self.clamp_log_sigma[1])

        sigma = torch.exp(log_sigma) + self.eps

        # 3) compute z
        z = (logit_x - mu) / sigma

        # 4) log Gaussian
        #    -0.5*z^2 - log(sigma) - log(sqrt(2*pi))
        log_gaussian = -0.5 * (z**2) - (torch.log(sigma) + self.LOG_SQRT_2PI)

        # 5) jacobian
        jacobian = -torch.log(x*(1 - x) + self.eps)

        log_prob = log_gaussian + jacobian
        nll = -log_prob

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll

class LogitStudentTFixedNuLoss(nn.Module):
    """
    Single-component Student's t on the logit scale, with a fixed nu > 0.

    Model output: (B,2,T) => [mu, log_sigma].
    x in [0,1].
    nu is a scalar float set in constructor.
    """
    def __init__(self, nu=3.0, eps=1e-6,
                 clamp_mu=(-10,10), clamp_log_sigma=(-10,10),
                 reduction='mean'):
        super().__init__()
        self.nu = float(nu)  # fixed degrees of freedom
        self.eps = eps
        self.clamp_mu = clamp_mu
        self.clamp_log_sigma = clamp_log_sigma
        self.reduction = reduction

        # Precompute constants that might be used
        # but we need device info if we do gammaln, so be careful in forward.

    def forward(self, y, x):
        B, C, T = y.shape
        assert C == 2, f"Expected 2 channels (mu, log_sigma), got {C}"

        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        x = torch.clamp(x, self.eps, 1.0 - self.eps)
        logit_x = torch.logit(x)

        mu        = torch.clamp(y[:, 0, :], self.clamp_mu[0], self.clamp_mu[1])
        log_sigma = torch.clamp(y[:, 1, :], self.clamp_log_sigma[0], self.clamp_log_sigma[1])
        sigma     = torch.exp(log_sigma) + self.eps

        nu = torch.tensor(self.nu, device=y.device)
        z = (logit_x - mu) / sigma

        # Student’s t formula:
        # log p_t(z) = ln Gamma((nu+1)/2) - ln[ sqrt(nu*pi)*Gamma(nu/2)*sigma ]
        #             - (nu+1)/2 * ln[1 + (z^2)/nu ]
        from torch.special import gammaln

        log_numer = gammaln(0.5*(nu + 1.0))
        log_denom = (
            gammaln(0.5*nu) +
            0.5*torch.log(nu * torch.tensor(math.pi, device=y.device)) +
            torch.log(sigma)
        )
        # broadcast shape: (B,T)
        log_t = log_numer - log_denom - 0.5*(nu + 1.0)*torch.log1p(z**2 / nu)

        jacobian = -torch.log(x*(1 - x) + self.eps)
        log_prob = log_t + jacobian
        nll = -log_prob

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll


class LogitStudentTAdaptiveNuLoss(nn.Module):
    """
    Single-component Student's t on the logit scale, with learned degrees of freedom.

    Model output: (B,3,T) => [mu, log_sigma, log_nu].
    x in [0,1].
    """
    def __init__(self, eps=1e-6,
                 clamp_mu=(-10,10), clamp_log_sigma=(-10,10), clamp_log_nu=(-3,10),
                 reduction='mean'):
        super().__init__()
        self.eps = eps
        self.clamp_mu = clamp_mu
        self.clamp_log_sigma = clamp_log_sigma
        self.clamp_log_nu = clamp_log_nu
        self.reduction = reduction

    def forward(self, y, x):
        B, C, T = y.shape
        assert C == 3, f"Expected 3 channels (mu, log_sigma, log_nu), got {C}"

        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        x = torch.clamp(x, self.eps, 1.0 - self.eps)
        logit_x = torch.logit(x)

        mu        = torch.clamp(y[:, 0, :], self.clamp_mu[0], self.clamp_mu[1])
        log_sigma = torch.clamp(y[:, 1, :], self.clamp_log_sigma[0], self.clamp_log_sigma[1])
        log_nu    = torch.clamp(y[:, 2, :], self.clamp_log_nu[0], self.clamp_log_nu[1])

        sigma = torch.exp(log_sigma) + self.eps
        nu    = torch.exp(log_nu) + self.eps

        z = (logit_x - mu) / sigma

        # log_numer = ln Gamma((nu+1)/2)
        # log_denom = ln Gamma(nu/2) + 0.5 ln(nu*pi) + ln(sigma)
        log_numer = gammaln(0.5*(nu + 1.0))
        log_denom = gammaln(0.5*nu) + 0.5*torch.log(nu*math.pi) + torch.log(sigma)

        log_t = log_numer - log_denom - 0.5*(nu + 1.0)*torch.log1p(z**2 / nu)

        jacobian = -torch.log(x*(1 - x) + self.eps)
        log_prob = log_t + jacobian
        nll = -log_prob

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll

class AlphaStableLoss(nn.Module):
    """
    Approximate negative log-likelihood for an alpha-stable distribution
    with parameters (alpha, beta, gamma, delta).

    x ~ S(alpha, beta, gamma, delta)

    The PDF has no closed form for general alpha < 2, so we use a
    numeric approximation method. This can be computationally expensive
    and less stable than standard distributions.

    Expected model output (if learning params for each sample/time):
      y: (B, 4, T) => [alpha, beta, log_gamma, delta]
      or if alpha,beta fixed, then fewer channels.

    x: (B, T) => real-valued data.

    Reference:
    - Nolan, J.P. (1997). Numerical computation of stable densities and distribution functions.
    - https://github.com/BartMassey/stable-pdf (example code in C).
    """

    def __init__(self,
                 approximate_method='fourier',
                 clamp_alpha=(0.1, 2.0),
                 clamp_beta=(-1.0, 1.0),
                 clamp_log_gamma=(-10, 10),
                 clamp_delta=(-100,100),
                 eps=1e-12,
                 reduction='mean'):
        """
        approximate_method: 'fourier' or 'series', indicates how we approximate PDF.
        clamp_*: clamp ranges for each parameter to avoid blowups.
        eps: small constant to avoid log(0).
        """
        super().__init__()
        self.approximate_method = approximate_method
        self.clamp_alpha = clamp_alpha
        self.clamp_beta = clamp_beta
        self.clamp_log_gamma = clamp_log_gamma
        self.clamp_delta = clamp_delta
        self.eps = eps
        self.reduction = reduction

    def forward(self, y, x):
        """
        y: (B, 4, T) => (alpha, beta, log_gamma, delta) or fewer channels if some fixed
        x: (B, T)
        Returns negative log-likelihood (scalar if reduction='mean').

        For each point x_{i}, we compute the approximate PDF under alpha-stable.
        """
        B, C, T = y.shape
        # If you fix alpha,beta in code, you'd have fewer channels. For demo, we assume 4 channels.
        assert C == 4, f"Expected 4 channels (alpha,beta,log_gamma,delta), got {C}"

        # 1) Extract parameters
        alpha = torch.clamp(y[:, 0, :], self.clamp_alpha[0], self.clamp_alpha[1])  # (B,T)
        beta  = torch.clamp(y[:, 1, :], self.clamp_beta[0], self.clamp_beta[1])
        log_gamma = torch.clamp(y[:, 2, :], self.clamp_log_gamma[0], self.clamp_log_gamma[1])
        delta     = torch.clamp(y[:, 3, :], self.clamp_delta[0], self.clamp_delta[1])

        gamma = torch.exp(log_gamma) + self.eps  # scale > 0

        # 2) We'll approximate pdf for each (alpha_i, beta_i, gamma_i, delta_i).
        #    We'll do so elementwise for each (B,T) to keep it simpler:
        x_reshaped = x.view(-1)    # flatten
        alpha_reshaped = alpha.view(-1)
        beta_reshaped  = beta.view(-1)
        gamma_reshaped = gamma.view(-1)
        delta_reshaped = delta.view(-1)

        pdf_vals = self._approx_stable_pdf(
            x_reshaped,
            alpha_reshaped,
            beta_reshaped,
            gamma_reshaped,
            delta_reshaped
        )

        # shape => (B*T,)
        pdf_vals = torch.clamp(pdf_vals, min=self.eps)

        log_pdf = torch.log(pdf_vals)
        nll = -log_pdf.view(B, T)  # (B,T)

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll  # (B,T)

    def _approx_stable_pdf(self, x, alpha, beta, gamma, delta):
        """
        Approximate the alpha-stable PDF at each point in x, given
        alpha,beta,gamma,delta (all shape=(N,) flattened).

        Returns pdf_vals of shape (N,).

        We'll do a simplified 'fourier' inversion approach or a 'series' approach.
        This is just a placeholder to show how you'd wire up the numeric approximation.
        In practice, you want a more robust library or carefully-coded routine.
        """
        if self.approximate_method == 'fourier':
            return self._fourier_inversion_pdf(x, alpha, beta, gamma, delta)
        else:
            return self._series_approx_pdf(x, alpha, beta, gamma, delta)

    def _fourier_inversion_pdf(self, x, alpha, beta, gamma, delta):
        """
        Example placeholder for an inverse Fourier transform approximation.
        Real code would discretize 't' and do a numerical integral of e^{-ixt} * phi_X(t).
        That is beyond the scope of a short snippet, so we'll stub it out as a
        demonstration returning a (N,) torch vector with a dummy formula.
        """
        N = x.shape[0]
        # Placeholder: Just return some function that has heavy tails, not a real stable pdf
        # purely to illustrate shape. This is *not* mathematically correct!
        # Use an actual library or your own detailed numeric code.
        # For example, we might fallback to cauchy if alpha=1 or so.

        # let's do a naive partial approach:
        # cauchy-like fallback
        z = (x - delta) / gamma
        # approximate cauchy if alpha ~ 1
        # pdf ~ 1 / [ pi * gamma * (1 + (z)^2 ) ]
        # Then maybe we add a small fudge factor if alpha < 1
        # This is purely illustrative.
        denom = math.pi * gamma * (1 + z**2)
        pdf_vals = 1.0 / denom
        # artificially scale tails if alpha < 1
        mask = (alpha < 1.0)
        pdf_vals[mask] = pdf_vals[mask] * 0.5  # total hack for demonstration
        return pdf_vals

    def _series_approx_pdf(self, x, alpha, beta, gamma, delta):
        """
        Another placeholder for a series expansion approximation.
        """
        # For demonstration, we replicate the same logic
        return self._fourier_inversion_pdf(x, alpha, beta, gamma, delta)
