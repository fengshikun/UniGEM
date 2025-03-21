from abc import abstractmethod, ABCMeta
from typing import Optional
import ase
from egnn.utils import act_class_mapping, GatedEquivariantBlock, GatedEquivariantBlockInject
from torch_scatter import scatter
import torch
from torch import nn
import numpy as np


__all__ = ["Scalar", "DipoleMoment", "ElectronicSpatialExtent", "MaskHead", "ElectronicSpatialExtent2", "VectorOutput2", "ScalarMol", "ScalarMolInject", "ScalarMol2", "ScalarMolInject2", "ScalarMolInject3", "VectorOutput3"]


class OutputModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, allow_prior_model):
        super(OutputModel, self).__init__()
        self.allow_prior_model = allow_prior_model

    def reset_parameters(self):
        pass

    @abstractmethod
    def pre_reduce(self, x, v, z, pos, batch):
        return

    def post_reduce(self, x):
        return x


class MaskHead(OutputModel):
    def __init__(self, hidden_channels, activation="silu", allow_prior_model=True, atom_num=119):
        super(MaskHead, self).__init__(allow_prior_model=allow_prior_model)
        act_class = act_class_mapping[activation]
        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            act_class(),
            nn.Linear(hidden_channels, atom_num),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_network[2].weight)
        self.output_network[2].bias.data.fill_(0)

    def pre_reduce(self, x):
        return self.output_network(x)


class Scalar(OutputModel):
    def __init__(self, hidden_channels, activation="silu", allow_prior_model=True):
        super(Scalar, self).__init__(allow_prior_model=allow_prior_model)
        act_class = act_class_mapping[activation]
        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            act_class(),
            nn.Linear(hidden_channels // 2, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_network[2].weight)
        self.output_network[2].bias.data.fill_(0)

    def pre_reduce(self, x, v: Optional[torch.Tensor], z, pos, batch):
        return self.output_network(x)


class EquivariantScalar(OutputModel):
    def __init__(self, hidden_channels, activation="silu", allow_prior_model=True):
        super(EquivariantScalar, self).__init__(allow_prior_model=allow_prior_model)
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(hidden_channels // 2, 1, activation=activation),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, v):
        for layer in self.output_network:
            x, v = layer(x, v)
        # include v in output to make sure all parameters have a gradient
        return x + v.sum() * 0
    

# equivariant scalar for the molnet, for multitask logits
class EquivariantScalarMol(OutputModel):
    def __init__(self, hidden_channels, output_channels, activation="silu", allow_prior_model=True):
        super(EquivariantScalarMol, self).__init__(allow_prior_model=allow_prior_model)
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(hidden_channels // 2, output_channels, activation=activation),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        # include v in output to make sure all parameters have a gradient
        
        
        output_atoms_logits = x + v.sum() * 0
        # batch scatter
        # output_logits = scatter(output_atoms_logits, batch, dim=0, reduce='mean')
        return output_atoms_logits

# normal head, only take x(mean) as input
class EquivariantScalarMol2(OutputModel):
    def __init__(self, hidden_channels, output_channels, activation="silu", allow_prior_model=True):
        super(EquivariantScalarMol2, self).__init__(allow_prior_model=allow_prior_model)
        act_class = act_class_mapping[activation]
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            act_class(),
            nn.Linear(hidden_channels, output_channels),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def pre_reduce(self, x, v, z, pos, batch):
        
        # scatter mean
        output_logits = scatter(x, batch, dim=0, reduce='mean')
        # output
        x = self.update_net(output_logits)
        
        output_atoms_logits = x + v.sum() * 0
        # batch scatter
        # output_logits = scatter(output_atoms_logits, batch, dim=0, reduce='mean')
        return output_atoms_logits

    

class EquivariantScalarMolInject2(OutputModel):
    def __init__(self, hidden_channels, output_channels, activation="silu", allow_prior_model=True, info_file='/data/protein/SKData/Pretraining-Denoising/bbbp_info_dict.pickle'):
        super(EquivariantScalarMolInject2, self).__init__(allow_prior_model=allow_prior_model)
        
        import pickle
        with open(info_file, 'rb') as handle:
            self.info_dict = pickle.load(handle)
            info_dims = len(self.info_dict['C(=O)(OC(C)(C)C)CCCc1ccc(cc1)N(CCCl)CCCl'])
            # norm:
            info_array = []
            for k in self.info_dict:
                info_array.append(self.info_dict[k])
            info_array = np.array(info_array)
            
            im = np.mean(info_array, axis=0)
            istd = np.std(info_array, axis=0)
            
            for k in self.info_dict:
                ele_info = self.info_dict[k]
                self.info_dict[k] = (np.array(ele_info) - im) / istd
            
            
        
        
        act_class = act_class_mapping[activation]
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels + info_dims, hidden_channels),
            act_class(),
            nn.Linear(hidden_channels, output_channels),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def pre_reduce(self, x, v, z, pos, batch, smi):
        info_array = []
        for sm in smi:
            info_array.append(self.info_dict[sm])
        
        # scatter mean
        output_logits = scatter(x, batch, dim=0, reduce='mean')
        # output
        inject_info = torch.tensor(info_array, dtype=x.dtype, device=x.device)
        output_logits = torch.cat([output_logits, inject_info], dim=-1)
        
        x = self.update_net(output_logits)
        
        
        output_atoms_logits = x + v.sum() * 0
        # batch scatter
        # output_logits = scatter(output_atoms_logits, batch, dim=0, reduce='mean')
        return output_atoms_logits



class EquivariantScalarMolInject3(OutputModel):
    def __init__(self, hidden_channels, output_channels, activation="silu", allow_prior_model=True, info_file='/data/protein/SKData/Pretraining-Denoising/bbbp_info_dict.pickle'):
        super(EquivariantScalarMolInject3, self).__init__(allow_prior_model=allow_prior_model)
        
        import pickle
        with open(info_file, 'rb') as handle:
            self.info_dict = pickle.load(handle)
            info_dims = len(self.info_dict['C(=O)(OC(C)(C)C)CCCc1ccc(cc1)N(CCCl)CCCl'])
            # norm:
            info_array = []
            for k in self.info_dict:
                info_array.append(self.info_dict[k])
            info_array = np.array(info_array)
            
            im = np.mean(info_array, axis=0)
            istd = np.std(info_array, axis=0)
            
            for k in self.info_dict:
                ele_info = self.info_dict[k]
                self.info_dict[k] = (np.array(ele_info) - im) / istd
            
            
        
        
        act_class = act_class_mapping[activation]
        self.update_net = nn.Sequential(
            nn.Linear(info_dims, hidden_channels),
            act_class(),
            nn.Linear(hidden_channels, output_channels),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def pre_reduce(self, x, v, z, pos, batch, smi):
        info_array = []
        for sm in smi:
            info_array.append(self.info_dict[sm])
        
        # scatter mean
        output_logits = scatter(x, batch, dim=0, reduce='mean')
        # output
        inject_info = torch.tensor(info_array, dtype=x.dtype, device=x.device)
        # output_logits = torch.cat([output_logits, inject_info], dim=-1)
        
        x = self.update_net(inject_info)
        
        
        output_atoms_logits = x + v.sum() * 0
        # batch scatter
        # output_logits = scatter(output_atoms_logits, batch, dim=0, reduce='mean')
        return output_atoms_logits


# experimental head for the bbbp(add the experience dimension)

class EquivariantScalarMolInject(OutputModel):
    def __init__(self, hidden_channels, output_channels, activation="silu", allow_prior_model=True, info_file='/data/protein/SKData/Pretraining-Denoising/bbbp_info_dict.pickle'):
        super(EquivariantScalarMolInject, self).__init__(allow_prior_model=allow_prior_model)
        
        import pickle
        with open(info_file, 'rb') as handle:
            self.info_dict = pickle.load(handle)
            info_dims = len(self.info_dict['C(=O)(OC(C)(C)C)CCCc1ccc(cc1)N(CCCl)CCCl'])
        
        
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlockInject(hidden_channels // 2, output_channels, activation=activation, inject_dim=info_dims),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, v, z, pos, batch, smi):
        info_array = []
        for sm in smi:
            info_array.append(self.info_dict[sm])
        
        for layer in self.output_network:
            if 'GatedEquivariantBlockInject' in str(type(layer)):
                x, v = layer(x, v, info_array, batch)
            else:
                x, v = layer(x, v)
        # include v in output to make sure all parameters have a gradient
        
        
        output_atoms_logits = x + v.sum() * 0
        # batch scatter
        # output_logits = scatter(output_atoms_logits, batch, dim=0, reduce='mean')
        return output_atoms_logits


class DipoleMoment(Scalar):
    def __init__(self, hidden_channels, activation="silu"):
        super(DipoleMoment, self).__init__(
            hidden_channels, activation, allow_prior_model=False
        )
        atomic_mass = torch.from_numpy(ase.data.atomic_masses).float()
        self.register_buffer("atomic_mass", atomic_mass)

    def pre_reduce(self, x, v: Optional[torch.Tensor], z, pos, batch):
        x = self.output_network(x)

        # Get center of mass.
        mass = self.atomic_mass[z].view(-1, 1)
        c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
        x = x * (pos - c[batch])
        return x

    def post_reduce(self, x):
        return torch.norm(x, dim=-1, keepdim=True)


class EquivariantDipoleMoment(EquivariantScalar):
    def __init__(self, hidden_channels, activation="silu"):
        super(EquivariantDipoleMoment, self).__init__(
            hidden_channels, activation, allow_prior_model=False
        )
        atomic_mass = torch.from_numpy(ase.data.atomic_masses).float()
        self.register_buffer("atomic_mass", atomic_mass)

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)

        # Get center of mass.
        mass = self.atomic_mass[z].view(-1, 1)
        c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
        x = x * (pos - c[batch])
        return x + v.squeeze()

    def post_reduce(self, x):
        return torch.norm(x, dim=-1, keepdim=True)


class ElectronicSpatialExtent(OutputModel):
    def __init__(self, hidden_channels, activation="silu"):
        super(ElectronicSpatialExtent, self).__init__(allow_prior_model=False)
        act_class = act_class_mapping[activation]
        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            act_class(),
            nn.Linear(hidden_channels // 2, 1),
        )
        atomic_mass = torch.from_numpy(ase.data.atomic_masses).float()
        self.register_buffer("atomic_mass", atomic_mass)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_network[2].weight)
        self.output_network[2].bias.data.fill_(0)

    def pre_reduce(self, x, v: Optional[torch.Tensor], z, pos, batch):
        x = self.output_network(x)

        # Get center of mass.
        mass = self.atomic_mass[z].view(-1, 1)
        c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)

        x = torch.norm(pos - c[batch], dim=1, keepdim=True) ** 2 * x
        return x




class EquivariantElectronicSpatialExtent2(EquivariantScalar):
    def __init__(self, hidden_channels, activation="silu"):
        super(EquivariantElectronicSpatialExtent2, self).__init__(hidden_channels, activation, allow_prior_model=False)
        # act_class = act_class_mapping[activation]
        # self.output_network = nn.Sequential(
        #     nn.Linear(hidden_channels, hidden_channels // 2),
        #     act_class(),
        #     nn.Linear(hidden_channels // 2, 1),
        # )
        atomic_mass = torch.from_numpy(ase.data.atomic_masses).float()
        self.register_buffer("atomic_mass", atomic_mass)

        # self.reset_parameters()

    # def reset_parameters(self):
    #     nn.init.xavier_uniform_(self.output_network[0].weight)
    #     self.output_network[0].bias.data.fill_(0)
    #     nn.init.xavier_uniform_(self.output_network[2].weight)
    #     self.output_network[2].bias.data.fill_(0)

    def pre_reduce(self, x, v: Optional[torch.Tensor], z, pos, batch):
        # x = self.output_network(x)
        for layer in self.output_network:
            x, v = layer(x, v)

        # Get center of mass.
        mass = self.atomic_mass[z].view(-1, 1)
        c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)

        x = torch.norm(pos - c[batch], dim=1, keepdim=True) ** 2 * x
        return x


class EquivariantElectronicSpatialExtent(ElectronicSpatialExtent):
    pass


class EquivariantVectorOutput(EquivariantScalar):
    def __init__(self, hidden_channels, activation="silu"):
        super(EquivariantVectorOutput, self).__init__(
            hidden_channels, activation, allow_prior_model=False
        )

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        return v.squeeze()


class EquivariantVectorOutput2(OutputModel):
    # def __init__(self, hidden_channels, activation="silu"):
    #     super(EquivariantVectorOutput, self).__init__(
    #         hidden_channels, activation, allow_prior_model=False
    #     )

    # def pre_reduce(self, x, v, z, pos, batch):
    #     for layer in self.output_network:
    #         x, v = layer(x, v)
    #     return v.squeeze()
    
    def __init__(self, hidden_channels, activation="silu", allow_prior_model=True):
        super(EquivariantVectorOutput2, self).__init__(allow_prior_model=allow_prior_model)
        act_class = act_class_mapping[activation]
        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            act_class(),
            nn.Linear(hidden_channels // 2, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_network[2].weight)
        self.output_network[2].bias.data.fill_(0)

    def pre_reduce(self, x, v: Optional[torch.Tensor], z, pos, batch):
        v = self.output_network(v)

        return v.squeeze()




# for both 3d denoise and 3d based cl
class EquivariantVectorOutput3(EquivariantScalar):
    def __init__(self, hidden_channels, activation="silu"):
        super(EquivariantVectorOutput3, self).__init__(
            hidden_channels, activation, allow_prior_model=False
        )
        self.ctl_3d = EquivariantCtlEmb(hidden_channels=hidden_channels, activation=activation, output_channels=hidden_channels)

    def pre_reduce(self, x, v, z, pos, batch):
        org_x = x
        org_v = v
        for layer in self.output_network:
            x, v = layer(x, v)
        
        ctl_pred = self.ctl_3d.pre_reduce(org_x, org_v, z, pos, batch)
        return v.squeeze(), ctl_pred, org_x

class EquivariantCtlEmb(OutputModel):
    def __init__(self, hidden_channels, activation="silu", allow_prior_model=True, output_channels=512):
        super(EquivariantCtlEmb, self).__init__(allow_prior_model=allow_prior_model)
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels,
                    activation=activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(hidden_channels, output_channels, activation=activation),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        # include v in output to make sure all parameters have a gradient
        return x + v.sum() * 0
