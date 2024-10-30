import torch
from typing import Optional
import math
import numpy as np
from jarvis.core.specie import chem_data, get_node_attributes

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda:6")



def get_attribute_lookup():
    max_z = max(v["Z"] for v in chem_data.values())
    template = get_node_attributes("C", atom_features='cgcnn')
    features = np.zeros((1 + max_z, len(template)))

    for element, v in chem_data.items():
        z = v["Z"]
        x = get_node_attributes(element, atom_features='cgcnn')
        #x = get_cgcnn_value(element,20) # 47 + 5N
        if x is not None:
            features[z, :] = x

    return features

FEATURE = get_attribute_lookup()
FEATURE = torch.from_numpy(FEATURE).to(dtype=torch.float32)
FEATURE = FEATURE.to(device)


class RBFExpansion(torch.nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )
        
        
class RBFExpansion_node(torch.nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(self):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        return FEATURE[ distance.squeeze().long() ]
        
         
        
        
class RBFExpansion_edge(torch.nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 64,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        self.gamma1 = 1/0.01
        self.gamma2 = 1/0.1 
        self.gamma3 = 1

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        # distance[:,2] = torch.norm(distance[:,2],dim=0)
        a1 = torch.exp(
            -self.gamma1 * (distance[:,2].unsqueeze(1) - self.centers) ** 2
        ).view(-1,self.bins)
        a2 = torch.exp(
            -self.gamma2 * (distance[:,2].unsqueeze(1) - self.centers) ** 2
        ).view(-1,self.bins)
        a3 = torch.exp(
            -self.gamma3 * (distance[:,2].unsqueeze(1) - self.centers) ** 2
        ).view(-1,self.bins)
        a6 = FEATURE[ distance[:,0].long() ]
        a7 = FEATURE[ distance[:,1].long() ]
        return torch.cat([a6,a7,a1,a2,a3],dim=1)
        

class RBFExpansion_triangle(torch.nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 64,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        self.gamma1 = 1/0.01
        self.gamma2 = 1/0.1 
        self.gamma3 = 1

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        a1 = torch.exp(
            -self.gamma1 * (distance[:,3].unsqueeze(1) - self.centers) ** 2
        ).view(-1,self.bins)
        a2 = torch.exp(
            -self.gamma2 * (distance[:,3].unsqueeze(1) - self.centers) ** 2
        ).view(-1,self.bins)
        a3 = torch.exp(
            -self.gamma3 * (distance[:,3].unsqueeze(1) - self.centers) ** 2
        ).view(-1,self.bins)
        a6 = FEATURE[ distance[:,0].long() ]
        a7 = FEATURE[ distance[:,1].long() ]
        a8 = FEATURE[ distance[:,2].long() ]
        return torch.cat([a6,a7,a8,a1,a2,a3],dim=1)
        
        

class RBFExpansion_triangle_dis(torch.nn.Module):
    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 64,
        rbf_function: str = "gaussian",
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.rbf_function = rbf_function

        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )
        self.lengthscale = np.diff(self.centers).mean()
        self.gamma = 1 / self.lengthscale

        self.gamma1 = 1/0.01
        self.gamma2 = 1/0.1 
        self.gamma3 = 1

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        a = []
        for i in range(9):
            a1 = self.gamma1 * (distance[:,i].unsqueeze(-1) - self.centers)
            a1 = (-a1 ** 2).exp()
            a2 = self.gamma2 * (distance[:,i].unsqueeze(-1) - self.centers)
            a2 = (-a2 ** 2).exp()
            a3 = self.gamma3 * (distance[:,i].unsqueeze(-1) - self.centers)
            a3 = (-a3 ** 2).exp()

            # a1 = self.gamma * (distance[:,i].unsqueeze(-1) - self.centers)
            # a1 = (-a1 ** 2).exp()

            a.append(a1)
            a.append(a2)
            a.append(a3)
        return torch.cat(a,dim=1)