import torch.nn as nn
import torch
from torch.distributions.multivariate_normal import MultivariateNormal


class GMLayer(nn.Module):
    def __init__(self, n_features, num_gaussians, device, requires_grad=True, num_classes=1, class_conditioned=False):
        super(GMLayer, self).__init__()
        self.n_features = n_features
        self.num_gaussians = num_gaussians
        self.device = device
        self.class_conditioned = class_conditioned
        self.num_classes = num_classes

        if class_conditioned:
            self.centers = nn.Parameter(
                torch.randn(num_classes, num_gaussians, n_features), requires_grad=requires_grad
            )
            cov = torch.eye(n_features).repeat(num_classes, num_gaussians, 1, 1)
            self.covariance = nn.Parameter(cov, requires_grad=requires_grad)
        else:
            self.centers = nn.Parameter(
                torch.randn(num_gaussians, n_features), requires_grad=requires_grad
            )
            cov = torch.eye(n_features).repeat(num_gaussians, 1, 1)
            self.covariance = nn.Parameter(cov, requires_grad=requires_grad)
    
    def compute_likelihoods(self, x):
        # x: [B, M, D]
        B, M, D = x.shape

        if self.class_conditioned:
            # Expand dimensions
            x_exp = x.unsqueeze(2).unsqueeze(2)                    # [B, M, 1, 1, D]
            centers = self.centers.unsqueeze(0).unsqueeze(0)       # [1, 1, C, K, D]
            covs = self.covariance.unsqueeze(0).unsqueeze(0)       # [1, 1, C, K, D, D]

            # Compute difference
            diff = x_exp - centers                                 # [B, M, C, K, D]
            cov_eps = 1e-5  # or 1e-4 if NaNs persist
            eye = torch.eye(D, device=covs.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,D,D]
            cov_reg = covs + cov_eps * eye  # broadcasted addition: [1,1,C,K,D,D]
            cov_inv = torch.inverse(cov_reg.squeeze(0).squeeze(0))  # [C, K, D, D]

            det_cov = torch.linalg.det(cov_reg.squeeze(0).squeeze(0)).clamp(min=1e-8) # [C, K]

            print("diff:", diff.shape)       # Debug: [B, M, C, K, D]
            print("cov_inv:", cov_inv.shape) # Debug: [C, K, D, D]

            # Mahalanobis distance
            mahalanobis = torch.einsum('bmcid,ckde,bmcje->bmck', diff, cov_inv, diff)
            mahalanobis = torch.clamp(mahalanobis, min=0.0, max=1e4)  # Clamp for stability

            # Log probability
            norm_term = torch.log((2 * torch.pi) ** D * det_cov)   # [C, K]
            norm_term = norm_term.unsqueeze(0).unsqueeze(0)        # [1, 1, C, K]
            log_probs = -0.5 * (mahalanobis + norm_term)
            log_probs = torch.clamp(log_probs, min=-30.0, max=30.0)  # Clamp for exp stability
            probs = torch.exp(log_probs)
            probs_sum = probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            probs = probs / probs_sum
            return probs.reshape(B, M, -1)  # [B, M, C*K]


        else:
            centers_expanded = self.centers.unsqueeze(0).unsqueeze(0)  # [1, 1, K, D]
            x_expanded = x.unsqueeze(2)                                # [B, M, 1, D]
            diff = x_expanded - centers_expanded                       # [B, M, K, D]

            cov_inv = torch.inverse(self.covariance)                   # [K, D, D]
            det_cov = torch.linalg.det(self.covariance)               # [K]

            mahalanobis = torch.einsum(
                '...i,...ij,...j->...',
                diff,
                cov_inv.unsqueeze(0).unsqueeze(0),
                diff
            )  # [B, M, K]
            mahalanobis = torch.clamp(mahalanobis, min=0.0, max=1e4)

            normalization_term = (torch.log((2 * torch.pi) ** D * det_cov)).unsqueeze(0).unsqueeze(0)
            log_probs = -0.5 * (mahalanobis + normalization_term)
            log_probs = torch.clamp(log_probs, min=-30.0, max=30.0)
            probs = torch.exp(log_probs)
            probs_sum = probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            probs = probs / probs_sum
            return probs


    def forward(self, x):
        return self.compute_likelihoods(x)
