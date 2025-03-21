from typing import Optional, Tuple
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot_orthogonal
from torch_scatter import scatter
from egnn.utils import (
    NeighborEmbedding,
    CosineCutoff,
    Distance,
    rbf_class_mapping,
    act_class_mapping,
)
from torch.nn.parameter import Parameter
from torch.nn import Linear

from egnn.feats import dist_emb, angle_emb, torsion_emb, xyz_to_dat
import torch.nn.functional as F


# init edge features

class EdgeFeatureInit(nn.Module):
    r"""init the edge feature of torchmd net"""
    def __init__(self, distance_exp, activation, num_radial, hidden_channels) -> None:
        super().__init__()
        self.distance_exp = distance_exp
        self.act = activation()
        self.lin_rbf_0 = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_1 = nn.Linear(num_radial, hidden_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rbf_0.reset_parameters()
        self.lin.reset_parameters()
        # self.lin_rbf_1.reset_parameters()
        glorot_orthogonal(self.lin_rbf_1.weight, scale=2.0)

    # TODO add 2D edge information
    def forward(self, node_embs, edge_index, edge_weight):
        # rbf: num_radial dimension
        rbf = self.distance_exp(edge_weight)
        # rbf0: latent dimension
        rbf0 = self.act(self.lin_rbf_0(rbf))
        # e1 is the first item in edge feature calculation
        # node_embs[edge_index[0]]: embedding of edge_index[0] atom type,
        # node_embs[edge_index[1]]: embedding of edge_index[1] atom type,
        # rbf0: distance embedding.
        e1 = self.act(self.lin(torch.cat([node_embs[edge_index[0]], node_embs[edge_index[1]], rbf0], dim=-1)))
        # rbf is the second item in edge feature calculation: (e^RBF)(dij)
        e2 = e1 * self.lin_rbf_1(rbf)
        return e1, e2


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act):
        super(ResidualLayer, self).__init__()
        self.act = act()
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))



class UpdateE(torch.nn.Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion, num_spherical, num_radial,
        num_before_skip, num_after_skip, act):
        super(UpdateE, self).__init__()
        self.act = act()
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size_dist, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size_dist, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size_angle, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size_angle, int_emb_size, bias=False)
        self.lin_t1 = nn.Linear(num_spherical * num_spherical * num_radial, basis_emb_size_torsion, bias=False)
        self.lin_t2 = nn.Linear(basis_emb_size_torsion, int_emb_size, bias=False)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_before_skip)
        ])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_after_skip)
        ])

        self.e0_norm = nn.LayerNorm(hidden_channels)
        self.e1_norm = nn.LayerNorm(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_t1.weight, scale=2.0)
        glorot_orthogonal(self.lin_t2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)

    def forward(self, x, emb, idx_kj, idx_ji):
        rbf0, sbf, t = emb
        x1,_ = x

        x_ji = self.act(self.lin_ji(x1))
        x_kj = self.act(self.lin_kj(x1))

        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))

        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        t = self.lin_t1(t)
        t = self.lin_t2(t)
        x_kj = x_kj * t

        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        e1 = x_ji + x_kj
        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin(e1)) + x1
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        e2 = self.lin_rbf(rbf0) * e1


        # NOTE layernorm for large value in e1 and e2
        e1 = self.e0_norm(e1)
        e2 = self.e1_norm(e2)

        return e1, e2


def check_for_nan(module, gin, gout):
    if gin[0].isnan().any():
        gin.isnan().any()
        print("NaN values found in gradients!")

class EMB(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent):
        super(EMB, self).__init__()
        self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent)
        self.angle_emb = angle_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.torsion_emb = torsion_emb(num_spherical, num_radial, cutoff, envelope_exponent)
        self.reset_parameters()

    def reset_parameters(self):
        self.dist_emb.reset_parameters()

    def forward(self, dist, angle, torsion, idx_kj):
        dist_emb = self.dist_emb(dist)
        angle_emb = self.angle_emb(dist, angle, idx_kj)
        torsion_emb = self.torsion_emb(dist, angle, torsion, idx_kj)
        return dist_emb, angle_emb, torsion_emb




# fuse 2D infomation inside
class TorchMD_ETF2D(nn.Module):
    r"""The TorchMD equivariant Transformer architecture.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_layers (int, optional): The number of attention layers.
            (default: :obj:`6`)
        num_rbf (int, optional): The number of radial basis functions :math:`\mu`.
            (default: :obj:`50`)
        rbf_type (string, optional): The type of radial basis function to use.
            (default: :obj:`"expnorm"`)
        trainable_rbf (bool, optional): Whether to train RBF parameters with
            backpropagation. (default: :obj:`True`)
        activation (string, optional): The type of activation function to use.
            (default: :obj:`"silu"`)
        attn_activation (string, optional): The type of activation function to use
            inside the attention mechanism. (default: :obj:`"silu"`)
        neighbor_embedding (bool, optional): Whether to perform an initial neighbor
            embedding step. (default: :obj:`True`)
        num_heads (int, optional): Number of attention heads.
            (default: :obj:`8`)
        distance_influence (string, optional): Where distance information is used inside
            the attention mechanism. (default: :obj:`"both"`)
        cutoff_lower (float, optional): Lower cutoff distance for interatomic interactions.
            (default: :obj:`0.0`)
        cutoff_upper (float, optional): Upper cutoff distance for interatomic interactions.
            (default: :obj:`5.0`)
        max_z (int, optional): Maximum atomic number. Used for initializing embeddings.
            (default: :obj:`100`)
        max_num_neighbors (int, optional): Maximum number of neighbors to return for a
            given node/atom when constructing the molecular graph during forward passes.
            This attribute is passed to the torch_cluster radius_graph routine keyword
            max_num_neighbors, which normally defaults to 32. Users should set this to
            higher values if they are using higher upper distance cutoffs and expect more
            than 32 neighbors per node/atom.
            (default: :obj:`32`)
    """

    def __init__(
        self,
        hidden_channels=128,
        num_layers=6,
        num_rbf=50,
        rbf_type="expnorm",
        trainable_rbf=True,
        activation="silu",
        attn_activation="silu",
        neighbor_embedding=True,
        num_heads=8,
        distance_influence="both",
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_z=100,
        max_num_neighbors=32,
        layernorm_on_vec=None,
        md17=False,
        seperate_noise=False,
        num_spherical=3, num_radial=6, envelope_exponent=5,
        int_emb_size=64,
        basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8,
        num_before_skip=1, num_after_skip=2, bond_pred=0,
    ):
        super(TorchMD_ETF2D, self).__init__()

        assert distance_influence in ["keys", "values", "both", "none"]
        assert rbf_type in rbf_class_mapping, (
            f'Unknown RBF type "{rbf_type}". '
            f'Choose from {", ".join(rbf_class_mapping.keys())}.'
        )
        assert activation in act_class_mapping, (
            f'Unknown activation function "{activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )
        assert attn_activation in act_class_mapping, (
            f'Unknown attention activation function "{attn_activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.neighbor_embedding = neighbor_embedding
        self.num_heads = num_heads
        self.distance_influence = distance_influence
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_z = max_z
        self.layernorm_on_vec = layernorm_on_vec

        act_class = act_class_mapping[activation]

        # if self.max_z > 200: # lba
        #     max_z = self.max_z // 2
        #     self.embedding = nn.Embedding(max_z, hidden_channels // 2)
        #     self.type_embedding = nn.Embedding(2, hidden_channels // 2)
        # else:
        #     self.embedding = nn.Embedding(self.max_z, hidden_channels)
        # self.embedding = nn.Embedding(self.max_z, hidden_channels)
        
        self.embedding = nn.Linear(max_z, hidden_channels)

        self.distance = Distance(
            cutoff_lower,
            cutoff_upper,
            max_num_neighbors=max_num_neighbors,
            return_vecs=True,
            loop=True,
        )
        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        self.neighbor_embedding = (
            NeighborEmbedding(
                hidden_channels, num_rbf, cutoff_lower, cutoff_upper, self.max_z
            ).jittable()
            if neighbor_embedding
            else None
        )

        self.attention_layers = nn.ModuleList()

        self.md17 = md17
        if not self.md17:
            self.vec_norms = nn.ModuleList()
            self.x_norms = nn.ModuleList()

        for _ in range(num_layers):
            layer = EquivariantMultiHeadAttention(
                hidden_channels,
                num_rbf,
                distance_influence,
                num_heads,
                act_class,
                attn_activation,
                cutoff_lower,
                cutoff_upper,
            ).jittable()
            self.attention_layers.append(layer)
            if not self.md17:
                self.vec_norms.append(EquivariantLayerNorm(hidden_channels))
                self.x_norms.append(nn.LayerNorm(hidden_channels))   # seems no use now

        self.out_norm = nn.LayerNorm(hidden_channels)

        self.seperate_noise = seperate_noise
        if self.seperate_noise:
            assert not self.layernorm_on_vec
            self.out_norm_vec = EquivariantLayerNorm(hidden_channels)




        self.init_e = EdgeFeatureInit(self.distance_expansion, act_class, num_rbf, hidden_channels)

        #  + 1
        self.update_es = torch.nn.ModuleList([
            UpdateE(hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion, num_spherical, num_radial, num_before_skip, num_after_skip, act_class) for _ in range(num_layers)])
        self.emb = EMB(num_spherical, num_radial, cutoff_upper, envelope_exponent)

        h = self.embedding.register_backward_hook(check_for_nan)



        if self.layernorm_on_vec:
            if self.layernorm_on_vec == "whitened":
                self.out_norm_vec = EquivariantLayerNorm(hidden_channels)
            else:
                raise ValueError(f"{self.layernorm_on_vec} not recognized.")
            
        
        self.T = 1000
        self.prediction_threshold_t = 10
        self.branch_layers_num = 8
        
        
        self.attention_layers2 = nn.ModuleList()
        self.vec_norms2 = nn.ModuleList()
        self.x_norm2 = nn.LayerNorm(hidden_channels, elementwise_affine=False)
        self.update_es2 = torch.nn.ModuleList()
        
        
        if self.branch_layers_num > 0:
            for i in range(0, self.branch_layers_num):
                layer = EquivariantMultiHeadAttention(
                    hidden_channels,
                    num_rbf,
                    distance_influence,
                    num_heads,
                    act_class,
                    attn_activation,
                    cutoff_lower,
                    cutoff_upper,
                ).jittable()
                self.attention_layers2.append(layer)
                self.vec_norms2.append(EquivariantLayerNorm(hidden_channels))
                
                self.update_es2.append(UpdateE(hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion, num_spherical, num_radial, num_before_skip, num_after_skip, act_class))

        self.bond_pred = bond_pred
        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        if self.neighbor_embedding is not None:
            self.neighbor_embedding.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        self.out_norm.reset_parameters()
        if self.layernorm_on_vec:
            self.out_norm_vec.reset_parameters()




    def forward(self, z, pos, batch, return_e=False, type_idx=None, half_batch_num=None):
        if self.bond_pred:
            return_e = True
        
        
        if self.branch_layers_num > 0:
            # we need split time out for the judgement of train/sampling branch layers
            # train: 10-1000(pre half), 10-10(later half)
            # sampling: all same time in one batch, only need to forward one pass
            all_main_branch = False
            all_second_branch = False
            
            # h_time schedule, TODO condition
            h_time = z[:, -1] * self.T 
            if (h_time > self.prediction_threshold_t).all():
                # sampling branch, all main
                all_main_branch = True
            elif (h_time < self.prediction_threshold_t).all():
                # sampling branch, all second
                all_second_branch = True
        
        
        x = self.embedding(z)
        if type_idx is not None:
            type_embedding = self.type_embedding(type_idx)
            x = torch.concat([x, type_embedding], dim=1)

        num_nodes=z.size(0)

        # edge_weight:distance, edge_vec:edge vector
        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        if self.branch_layers_num > 0:
            org_edge_index = edge_index

        # init edge feature
        e = self.init_e(x, edge_index, edge_weight) # contains e1 and e2


        # mask the self loop edge
        mask = edge_index[0] != edge_index[1]
        no_loop_edge_index = edge_index[:, mask]


        dist, angle, torsion, i, j, idx_kj, idx_ji = xyz_to_dat(pos, no_loop_edge_index, num_nodes, use_torsion=True)
        
        
        emb = self.emb(dist, angle, torsion, idx_kj)
        org_emb = (emb[0].clone(),emb[1].clone(),emb[2].clone())


        assert (
            edge_vec is not None
        ), "Distance module did not return directional information"

        edge_attr = self.distance_expansion(edge_weight) # replace with f2d
        mask = edge_index[0] != edge_index[1]
        # normalize the distance
        edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask], dim=1).unsqueeze(1)


        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)

        
        
        # for pos2
        if self.branch_layers_num > 0 and half_batch_num > 0:
            pos1 = pos[half_batch_num:]
            batch1 = batch[half_batch_num:]
            batch1 = batch1 - batch1.min()
            
            
            
            pos0 = pos[:half_batch_num]
            batch0 = batch[:half_batch_num]
            
            edge_index0, edge_weight0, edge_vec0 = self.distance(pos0, batch0)
            assert (
                edge_vec0 is not None
            ), "Distance module did not return directional information"

            edge_attr0 = self.distance_expansion(edge_weight0)
            mask0 = edge_index0[0] != edge_index0[1]
            edge_vec0[mask0] = edge_vec0[mask0] / torch.norm(edge_vec0[mask0], dim=1).unsqueeze(1)
            
            no_loop_edge_index0 = edge_index0[:, mask0]
            num_nodes0 = half_batch_num
            
            half_edge_num = edge_index0.size(1)
            e0 = (e[0][:half_edge_num], e[1][:half_edge_num])
            dist0, angle0, torsion0, i0, j0, idx_kj0, idx_ji0 = xyz_to_dat(pos0, no_loop_edge_index0, num_nodes0, use_torsion=True)
            emb0 = self.emb(dist0, angle0, torsion0, idx_kj0)
            
            
            
            edge_index1, edge_weight1, edge_vec1 = self.distance(pos1, batch1)
            assert (
                edge_vec1 is not None
            ), "Distance module did not return directional information"

            edge_attr1 = self.distance_expansion(edge_weight1)
            mask1 = edge_index1[0] != edge_index1[1]
            edge_vec1[mask1] = edge_vec1[mask1] / torch.norm(edge_vec1[mask1], dim=1).unsqueeze(1)
            e1 = (e[0][half_edge_num:], e[1][half_edge_num:])
            no_loop_edge_index1 = edge_index1[:, mask1]
            num_nodes1 = num_nodes - half_batch_num
            dist1, angle1, torsion1, i1, j1, idx_kj1, idx_ji1 = xyz_to_dat(pos1, no_loop_edge_index1, num_nodes1, use_torsion=True)
            emb1 = self.emb(dist1, angle1, torsion1, idx_kj1)
        
        # Update
        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)
        update_e0, epdate_e1 = e[0][mask], e[1][mask]
        for lidx, attn in enumerate(self.attention_layers):

            # update edge feature
            # update e without self loop
            update_e0, epdate_e1 = self.update_es[lidx]((update_e0, epdate_e1), emb, idx_kj, idx_ji)
            # replace edge_attr with the edge feature
            # NOTE: use e2 to update node feature
            edge_e1 = torch.clone(e[1]) # This function is differentiable, so gradients will flow back from the result of this operation to e[1]
            edge_e1[mask] = epdate_e1

            x_before_attn = x.clone()
            vec_before_attn = vec.clone()

            # seems like: x:node emb, vec:edge emb, edge_index:edge_index, edge_weight:r_ij, edge_e1:f_ij, edge_vec:d_ij. f_ij is the r_ij after RBF
            dx, dvec = attn(x, vec, edge_index, edge_weight, edge_e1, edge_vec)
            # if not self.md17:
            #     dx = self.x_norms[lidx](dx)
            x = x + dx # may be nan
            # if not self.md17:
            #     x = self.x_norms[lidx](x)
            # if not self.md17:
            #     dvec = self.vec_norms[lidx](dvec)
            vec = vec + dvec
            if not self.md17:
                vec = self.vec_norms[lidx](vec)
            if torch.isnan(x).sum():
                print('nan happens1111')
            
            if self.branch_layers_num > 0 and lidx == self.num_layers - self.branch_layers_num - 1 and not all_main_branch:
                if all_second_branch: # all the second branch, but the first branch is also forwarded, may be optimized further
                    x1 = x
                    vec1 = vec
                    edge_index1 = edge_index
                    edge_weight1 = edge_weight
                    edge_attr1 = edge_attr
                    edge_vec1 = edge_vec
                    
                    e1 = e
                    update_e01, epdate_e11 = update_e0, epdate_e1
                    mask1 = mask
                    
                    emb1 = emb
                    idx_kj1 = idx_kj
                    idx_ji1 = idx_ji
                else:
                    x1 = x[half_batch_num:]
                    vec1 = vec[half_batch_num:]
                    
                    
                    x = x[:half_batch_num]
                    vec = vec[:half_batch_num]
                    edge_index = edge_index0
                    edge_weight = edge_weight0
                    edge_attr = edge_attr0
                    edge_vec = edge_vec0
                    
                    emb = emb0
                    idx_kj = idx_kj0
                    idx_ji = idx_ji0
                    # update e
                    e1 = e1
                    mask_half_edge_num = mask0.sum()
                    update_e01, epdate_e11 = update_e0[mask_half_edge_num:], epdate_e1[mask_half_edge_num:]
                    
                    e = e0
                    update_e0, epdate_e1 = update_e0[:mask_half_edge_num], epdate_e1[:mask_half_edge_num]
                    mask = mask0
                    
                    
                    
            
            if self.branch_layers_num > 0 and lidx >= self.num_layers - self.branch_layers_num and not all_main_branch:
                idx = lidx - (self.num_layers - self.branch_layers_num)
                
                update_e01, epdate_e11 = self.update_es2[idx]((update_e01, epdate_e11), emb1, idx_kj1, idx_ji1)
                # replace edge_attr with the edge feature
                # NOTE: use e2 to update node feature
                edge_e11 = torch.clone(e1[1]) # This function is differentiable, so gradients will flow back from the result of this operation to e[1]
                edge_e11[mask1] = epdate_e11

                
                dx1, dvec1 = self.attention_layers2[idx](x1, vec1, edge_index1, edge_weight1, edge_e11, edge_vec1)
                x1 = x1 + dx1 # may be nan
                if not self.md17:
                    x1 = self.x_norm2(x1)
                vec1 = vec1 + dvec1
                if not self.md17:
                    vec1 = self.vec_norms2[idx](vec1)
        
        if self.branch_layers_num > 0 and not all_main_branch:
            if all_second_branch:
                x = x1
                vec = vec1
            else:
                x = torch.cat([x, x1], dim=0)
                vec = torch.cat([vec, vec1], dim=0)    
            
        if torch.isnan(x).sum():
            print('nan happens11112222')
        # x = torch.clip(x, min=-1e+7, max=1e+7)
        ## self.out_norm: output network
        xnew = self.out_norm(x)
        if torch.isnan(xnew).sum():
            print('nan happens111122223333')
            # import pdb; pdb.set_trace()# print('nan happens2222')
        if self.layernorm_on_vec:
            vec = self.out_norm_vec(vec)
        if self.seperate_noise:
            nvec = self.out_norm_vec(vec)
            return xnew, vec, nvec, z, pos, batch

        if return_e:
            if self.branch_layers_num > 0 and not all_main_branch:
                if all_second_branch:
                    # if epdate_e11.size(0) == 0:
                    #     print('nan happens1111222233334444')
                    return xnew, vec, z, pos, batch, epdate_e11, edge_index1[:,mask1]
                else:
                    # concat
                    update_e1 = torch.cat([epdate_e1, epdate_e11], dim=0)
                    mask = org_edge_index[0] != org_edge_index[1]
                    # if update_e1.size(0) == 0:
                    #     print('nan happens1111222233334444')
                    return xnew, vec, z, pos, batch, update_e1, org_edge_index[:,mask]
            
            e_clone_0 = torch.clone(e[0]) # clone is differentiable
            e_clone_1 = torch.clone(e[1])
            e_clone_0[mask], e_clone_1[mask] = update_e0, epdate_e1
            # if epdate_e1.size(0) == 0:
            #     print('nan happens1111222233334444')
            return xnew, vec, z, pos, batch, epdate_e1, edge_index[:,mask]

        
        
        return xnew, vec, z, pos, batch

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"rbf_type={self.rbf_type}, "
            f"trainable_rbf={self.trainable_rbf}, "
            f"activation={self.activation}, "
            f"attn_activation={self.attn_activation}, "
            f"neighbor_embedding={self.neighbor_embedding}, "
            f"num_heads={self.num_heads}, "
            f"distance_influence={self.distance_influence}, "
            f"cutoff_lower={self.cutoff_lower}, "
            f"cutoff_upper={self.cutoff_upper})"
        )


class EquivariantMultiHeadAttention(MessagePassing):
    def __init__(
        self,
        hidden_channels,
        num_rbf,
        distance_influence,
        num_heads,
        activation,
        attn_activation,
        cutoff_lower,
        cutoff_upper,
    ):
        super(EquivariantMultiHeadAttention, self).__init__(aggr="add", node_dim=0)
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.act = activation()
        self.attn_activation = act_class_mapping[attn_activation]()
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)

        self.dk_proj = None
        if distance_influence in ["keys", "both"]:
            # self.dk_proj = nn.Linear(num_rbf, hidden_channels)
            self.dk_proj = nn.Linear(hidden_channels, hidden_channels)

        self.dv_proj = None
        if distance_influence in ["values", "both"]:
            # self.dv_proj = nn.Linear(num_rbf, hidden_channels * 3)
            self.dv_proj = nn.Linear(hidden_channels, hidden_channels * 3)

        self.reset_parameters()

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.vec_proj.weight)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij):  # f_ij is the r_ij after RBF
        x = self.layernorm(x)
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim * 3)

        # vec1, vec2, vec3 is the U1, U2, U3
        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec = vec.reshape(-1, 3, self.num_heads, self.head_dim)
        # vec_dot is used in node update
        vec_dot = (vec1 * vec2).sum(dim=1)

        dk = (
            self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
            if self.dk_proj is not None
            else None
        )
        dv = (
            self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim * 3)
            if self.dv_proj is not None
            else None
        )

        # The transformer calculation, x is the yi
        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, vec: Tensor, dk: Tensor, dv: Tensor, r_ij: Tensor, d_ij: Tensor)
        x, vec = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            vec=vec,
            dk=dk,
            dv=dv,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        x = x.reshape(-1, self.hidden_channels)
        vec = vec.reshape(-1, 3, self.hidden_channels)

        # yi from attention first expand 3 times, then split to three parts. o1 used in vector update. o2 and o3 used in node update.
        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + o3

        # vec3 is U3. vec is from propagate, and it combine the sij1, sij2, rij, dij(ri - rj) to get the vec to add to vec3
        dvec = vec3 * o1.unsqueeze(1) + vec
        return dx, dvec

    def message(self, q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij):
        # attention mechanism
        if dk is None:
            attn = (q_i * k_j).sum(dim=-1)
        else:
            attn = (q_i * k_j * dk).sum(dim=-1)

        # attention activation function
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        # value pathway
        if dv is not None:
            v_j = v_j * dv
        x, vec1, vec2 = torch.split(v_j, self.head_dim, dim=2)

        # update scalar features
        # x = x * attn.unsqueeze(2)
        x = x * F.softmax(attn, dim=-1).unsqueeze(2)
        # update vector features
        vec = vec_j * vec1.unsqueeze(1) + vec2.unsqueeze(1) * d_ij.unsqueeze(
            2
        ).unsqueeze(3)
        return x, vec

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')
        return x, vec

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class EquivariantLayerNorm(nn.Module):
    r"""Rotationally-equivariant Vector Layer Normalization
    Expects inputs with shape (N, n, d), where N is batch size, n is vector dimension, d is width/number of vectors.
    """
    __constants__ = ["normalized_shape", "elementwise_linear"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_linear: bool

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_linear: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(EquivariantLayerNorm, self).__init__()

        self.normalized_shape = (int(normalized_shape),)
        self.eps = eps
        self.elementwise_linear = elementwise_linear
        if self.elementwise_linear:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None) # Without bias term to preserve equivariance!

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_linear:
            nn.init.ones_(self.weight)

    def mean_center(self, input):
        return input - input.mean(-1, keepdim=True)

    def covariance(self, input):
        return 1 / self.normalized_shape[0] * input @ input.transpose(-1, -2)

    def symsqrtinv(self, matrix):
        """Compute the inverse square root of a positive definite matrix.

        Based on https://github.com/pytorch/pytorch/issues/25481
        """
        _, s, v = matrix.svd()
        good = (
            s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
        )
        components = good.sum(-1)
        common = components.max()
        unbalanced = common != components.min()
        if common < s.size(-1):
            s = s[..., :common]
            v = v[..., :common]
            if unbalanced:
                good = good[..., :common]
        if unbalanced:
            s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
        return (v * 1 / torch.sqrt(s + self.eps).unsqueeze(-2)) @ v.transpose(
            -2, -1
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(torch.float64) # Need double precision for accurate inversion.
        input = self.mean_center(input)
        # We use different diagonal elements in case input matrix is approximately zero,
        # in which case all singular values are equal which is problematic for backprop.
        # See e.g. https://pytorch.org/docs/stable/generated/torch.svd.html
        reg_matrix = (
            torch.diag(torch.tensor([1.0, 2.0, 3.0]))
            .unsqueeze(0)
            .to(input.device)
            .type(input.dtype)
        )
        covar = self.covariance(input) + self.eps * reg_matrix
        covar_sqrtinv = self.symsqrtinv(covar)
        return (covar_sqrtinv @ input).to(
            self.weight.dtype
        ) * self.weight.reshape(1, 1, self.normalized_shape[0])

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, "
            "elementwise_linear={elementwise_linear}".format(**self.__dict__)
        )
