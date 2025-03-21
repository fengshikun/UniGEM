import torch


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().sum().item() < 1e-8


def compute_loss_and_nll(args, generative_model, nodes_dist, x, h, node_mask, edge_mask, context, uni_diffusion=False, mask_indicator=None, expand_diff=False, property_label=None, bond_info=None):
    bs, n_nodes, n_dims = x.size()


    if args.probabilistic_model == 'diffusion':
        edge_mask = edge_mask.view(bs, n_nodes * n_nodes)

        assert_correctly_masked(x, node_mask)

        # Here x is a position tensor, and h is a dictionary with keys
        # 'categorical' and 'integer'.
        
        
        if uni_diffusion:
            nll, loss_dict = generative_model(x, h, node_mask, edge_mask, context, mask_indicator=mask_indicator)
        else:
            nll, loss_dict = generative_model(x, h, node_mask, edge_mask, context, mask_indicator=mask_indicator, expand_diff=args.expand_diff, property_label=property_label, bond_info=bond_info)

        if args.bfn_schedule:
            return nll, torch.tensor([0], device=nll.device), torch.tensor([0], device=nll.device), loss_dict

        N = node_mask.squeeze(2).sum(1).long()

        log_pN = nodes_dist.log_prob(N)

        assert nll.size() == log_pN.size()
        nll = nll - log_pN

        # Average over batch.
        nll = nll.mean(0)

        reg_term = torch.tensor([0.]).to(nll.device)
        mean_abs_z = 0.
    else:
        raise ValueError(args.probabilistic_model)

    
    return nll, reg_term, mean_abs_z, loss_dict
    
    # if uni_diffusion:
    #     return nll, reg_term, mean_abs_z, loss_dict
    
    # return nll, reg_term, mean_abs_z
