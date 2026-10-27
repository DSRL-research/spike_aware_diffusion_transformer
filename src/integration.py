import numpy as np
import torch
from sklearn.mixture import GaussianMixture

KAPPAS = tuple(np.round(np.arange(0.0, 0.91, 0.1), 2))
BETAS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


def s1_shift(p: np.ndarray, kappa: float, beta: float) -> np.ndarray:
    return beta * p * (p >= kappa)


class GMMSpikeProbs:
    def __init__(self, probs: np.ndarray, fit_windows: np.ndarray, seed: int = 0, n_components: int = 2):
        logits = np.log(probs[fit_windows]) - np.log1p(-probs[fit_windows])
        logits = logits[np.isfinite(logits)].reshape(-1, 1)
        self.gmm = GaussianMixture(n_components, random_state=seed).fit(logits)
        self.seed = seed
        self.horizon = probs.shape[1]

    def draw(self, window: int) -> np.ndarray:
        rng = np.random.default_rng([self.seed, int(window)])
        comp = rng.choice(len(self.gmm.weights_), size=self.horizon, p=self.gmm.weights_)
        mean, sd = self.gmm.means_[comp, 0], np.sqrt(self.gmm.covariances_.reshape(-1)[comp])
        return 1.0 / (1.0 + np.exp(-(mean + sd * rng.standard_normal(self.horizon))))

    def table(self, windows) -> np.ndarray:
        return np.stack([self.draw(w) for w in windows])

    def __call__(self, batch: dict) -> torch.Tensor:
        p = self.table(batch["window"].tolist())
        return torch.tensor(p, dtype=torch.float32, device=batch["window"].device).unsqueeze(-1)
