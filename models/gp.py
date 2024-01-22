from typing import Any, Callable, List, Optional

import botorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.posteriors import Posterior
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel, SpectralMixtureKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from torch import Tensor


class SingleTaskGP(Model):

    def __init__(self, model_args, input_dim, output_dim):
        super().__init__()
        self.gp = None
        self.output_dim = output_dim
        self.nu = model_args["nu"] if "nu" in model_args else 2.5

        if "kernel" not in model_args:
            self.covar_module = None
        elif model_args["kernel"] == "Matern":
            self.covar_module = ScaleKernel(
                    MaternKernel(
                        nu=2.5,
                        ard_num_dims=input_dim,
                    )
                )
        elif model_args["kernel"] == "SpectralMixture-4" or model_args== "SpectralMixture":
            self.covar_module = SpectralMixtureKernel(
                num_mixtures=4, 
                ard_num_dims=input_dim)
        elif model_args["kernel"]== "SpectralMixture-10":
            self.covar_module = SpectralMixtureKernel(
                num_mixtures=10, 
                ard_num_dims=input_dim)
        elif model_args["kernel"]== "SpectralMixture-20":
            self.covar_module = SpectralMixtureKernel(
                num_mixtures=20, 
                ard_num_dims=input_dim)        
        else:
            print("Not a valid kernel") #should also throw error

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:
        return self.gp.posterior(X, output_indices, observation_noise, posterior_transform, **kwargs)

    @property
    def batch_shape(self) -> torch.Size:
        return self.gp.batch_shape

    @property
    def num_outputs(self) -> int:
        return self.gp.num_outputs

    def fit_and_save(self, train_x, train_y, save_dir):
        if self.output_dim > 1:
            raise RuntimeError(
                "SingleTaskGP does not fit tasks with multiple objectives")
        
        self.gp = botorch.models.gp_regression.SingleTaskGP(
            train_x, 
            train_y, 
            outcome_transform=Standardize(m=1),
            ).to(train_x)
        #print("covar_module:", self.gp.covar_module)
        if self.covar_module is not None:
            self.gp.covar_module = self.covar_module
        #print("covar_module:", self.gp.covar_module)
        
        mll = ExactMarginalLogLikelihood(
            self.gp.likelihood, self.gp).to(train_x)
        fit_gpytorch_mll(mll)


class MultiTaskGP(Model):

    def __init__(self, model_args, input_dim, output_dim):
        super().__init__()
        self.gp = None
        self.output_dim = output_dim
        self.nu = model_args["nu"] if "nu" in model_args else 2.5

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:
        return self.gp.posterior(X, output_indices, observation_noise, posterior_transform, **kwargs)

    @property
    def batch_shape(self) -> torch.Size:
        return self.gp.batch_shape

    @property
    def num_outputs(self) -> int:
        return self.gp.num_outputs

    def fit_and_save(self, train_x, train_y, save_dir):
        models = []
        for d in range(self.output_dim):
            models.append(
                botorch.models.SingleTaskGP(
                    train_x,
                    train_y[:, d].unsqueeze(-1),
                    outcome_transform=Standardize(m=1)).to(train_x)
                )

        self.gp = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(self.gp.likelihood, self.gp).to(train_x)
        fit_gpytorch_mll(mll)