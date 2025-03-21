# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import utils
import argparse
from qm9 import dataset
from qm9.models import get_model
import os
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
import torch
import time
import pickle
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9.sampling import sample, charge_decode
from qm9.analyze import analyze_stability_for_molecules, analyze_node_distribution
from qm9.utils import prepare_context, compute_mean_mad
from qm9 import visualizer as qm9_visualizer
import qm9.losses as losses

from train_test import save_and_sample_chain_save

try:
    from qm9 import rdkit_functions
except ModuleNotFoundError:
    print('Not importing rdkit functions.')


def check_mask_correct(variables, node_mask):
    for variable in variables:
        assert_correctly_masked(variable, node_mask)


def analyze_and_save(args, eval_args, device, generative_model,
                     nodes_dist, prop_dist, dataset_info, n_samples=10,
                     batch_size=10, save_to_xyz=False, sample_steps=1000):
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    if args.bfn_schedule:
        molecules = []
    else:
        molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    start_time = time.time()
    
    # test
    # n_samples=4
    # batch_size=2
    if eval_args.only_vis:
        args.exp_name = eval_args.vis_dir
        for i in range(n_samples):
            save_and_sample_chain_save(generative_model, args, device, dataset_info, prop_dist=None, batch_id=str(i))
            if i % 10 == 0:
                print(f'{i}/{n_samples} Molecules generated \n')
        exit()
    
    if hasattr(args, 'bond_pred') and args.bond_pred:
        bond_lst_all = []
    else:
        bond_lst_all = None
    
    for i in range(int(n_samples/batch_size)):
        # if i > 3:
        #     break
        nodesxsample = nodes_dist.sample(batch_size)
        if args.bfn_schedule:
            theta_traj, segment_ids = sample(args, device, generative_model, dataset_info, prop_dist,
                                                nodesxsample=nodesxsample, sample_steps=sample_steps)
        elif args.property_pred:
            one_hot, charges, x, node_mask, pred = sample(
                args, device, generative_model, dataset_info, prop_dist=prop_dist, nodesxsample=nodesxsample)
        else:
            one_hot, charges, x, node_mask = sample(
            args, device, generative_model, dataset_info, prop_dist=prop_dist, nodesxsample=nodesxsample)

        if not args.bfn_schedule and isinstance(x, list):
            x, bond_lst = x[0], x[1]
            bond_lst_all.extend(bond_lst)
        else:
            bond_lst = None
        
        if args.bfn_schedule:
            segment_ids = segment_ids.cpu()
            x, h = theta_traj[-1] # x: N x 3
            if h.shape[-1] == 1:
                atom_types = charge_decode(h, dataset_info) # N x 1 --> N x 5
            else:
                atom_types = h #  atom type prediction: h is N x 5 logits
            bz = segment_ids.max().item()
            
            # for saving xyz
            one_hot = []
            charges = []
            x_lst = []
            node_mask = None
            bond_lst = None
            
            for i in range(bz):
                pos = x[segment_ids == i].detach().cpu()
                sub_atom_type = atom_types[segment_ids == i] # n x 5
                sub_atom_type = sub_atom_type.argmax(1).cpu()
                molecules.append((pos, sub_atom_type))
                
                # change sub_atom_type to one hot
                one_hot.append(torch.eye(len(dataset_info['atom_encoder']))[sub_atom_type])
                # one_hot.append(sub_atom_type)
                x_lst.append(pos)
            x = x_lst
        else:
            molecules['one_hot'].append(one_hot.detach().cpu())
            molecules['x'].append(x.detach().cpu())
            molecules['node_mask'].append(node_mask.detach().cpu())

        current_num_samples = (i+1) * batch_size
        secs_per_sample = (time.time() - start_time) / current_num_samples
        print('\t %d/%d Molecules generated at %.2f secs/sample' % (
            current_num_samples, n_samples, secs_per_sample))

        if save_to_xyz:
            id_from = i * batch_size
            qm9_visualizer.save_xyz_file(
                join(eval_args.model_path, 'eval/analyzed_molecules/'),
                one_hot, charges, x, dataset_info, id_from, name='molecule',
                node_mask=node_mask, bond_info=bond_lst, bfn_schedule=args.bfn_schedule)
    if not args.bfn_schedule:
        molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    stability_dict, rdkit_metrics = analyze_stability_for_molecules(
        molecules, dataset_info, bfn_schedule=args.bfn_schedule, bond_lst=bond_lst_all)

    return stability_dict, rdkit_metrics


def test(args, flow_dp, nodes_dist, device, dtype, loader, partition='Test', num_passes=1):
    flow_dp.eval()
    nll_epoch = 0
    n_samples = 0
    for pass_number in range(num_passes):
        with torch.no_grad():
            for i, data in enumerate(loader):
                # Get data
                x = data['positions'].to(device, dtype)
                node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
                edge_mask = data['edge_mask'].to(device, dtype)
                one_hot = data['one_hot'].to(device, dtype)
                charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

                batch_size = x.size(0)

                x = remove_mean_with_mask(x, node_mask)
                check_mask_correct([x, one_hot], node_mask)
                assert_mean_zero_with_mask(x, node_mask)

                h = {'categorical': one_hot, 'integer': charges}

                if len(args.conditioning) > 0:
                    context = prepare_context(args.conditioning, data).to(device, dtype)
                    assert_correctly_masked(context, node_mask)
                else:
                    context = None

                # transform batch through flow
                nll, _, _, loss_dict = losses.compute_loss_and_nll(args, flow_dp, nodes_dist, x, h, node_mask, edge_mask, context, property_label=data[args.target_property].to(device, dtype) if args.property_pred else None)
                # standard nll from forward KL

                nll_epoch += nll.item() * batch_size
                n_samples += batch_size
                if i % args.n_report_steps == 0:
                    print(f"\r {partition} NLL \t, iter: {i}/{len(loader)}, "
                          f"NLL: {nll_epoch/n_samples:.2f}")

    return nll_epoch/n_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="outputs/edm_1",
                        help='Specify model path')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Specify model path')
    parser.add_argument('--batch_size_gen', type=int, default=100,
                        help='Specify model path')
    parser.add_argument('--save_to_xyz', type=eval, default=False,
                        help='Should save samples to xyz files.')
    parser.add_argument("--checkpoint_epoch", type=int, default=None,)
    parser.add_argument("--sample_steps", type=int, default=1000)
    parser.add_argument("--bfn_schedule", type=int, default=0)
    parser.add_argument("--bfn_str", type=int, default=0)
    parser.add_argument("--optimal_sampling", type=int, default=0)
    parser.add_argument('--only_vis', type=eval, default=False,) # only save vis results
    parser.add_argument('--vis_dir', type=str, default='vis_results') # only save vis results

    eval_args, unparsed_args = parser.parse_known_args()

    assert eval_args.model_path is not None

    with open(join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

        # args.branch_layers_num = 8
    # CAREFUL with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'

    if not hasattr(args, 'bfn_schedule'):
        args.bfn_schedule = eval_args.bfn_schedule

    if not hasattr(args, "bfn_str"):
        args.bfn_str = eval_args.bfn_str
    
    if not hasattr(args, "optimal_sampling"):
        args.optimal_sampling = eval_args.optimal_sampling
        
    # args.branch_layers_num = 8

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    dtype = torch.float32
    utils.create_folders(args)
    print(args)

    # Retrieve QM9 dataloaders
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    # Load model
    generative_model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
    if prop_dist is not None:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        prop_dist.set_normalizer(property_norms)
    generative_model.to(device)

    if eval_args.checkpoint_epoch is not None:
        fn = f"generative_model_ema_{eval_args.checkpoint_epoch}.npy" if args.ema_decay > 0 else f"generative_model_{eval_args.checkpoint_epoch}.npy"
    else:
        fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    print(f"Loading {fn} from {eval_args.model_path}")
    flow_state_dict = torch.load(join(eval_args.model_path, fn), map_location=device)
    generative_model.load_state_dict(flow_state_dict)

    # Analyze stability, validity, uniqueness and novelty
    stability_dict, rdkit_metrics = analyze_and_save(
        args, eval_args, device, generative_model, nodes_dist,
        prop_dist, dataset_info, n_samples=eval_args.n_samples,
        batch_size=eval_args.batch_size_gen, save_to_xyz=eval_args.save_to_xyz, sample_steps=eval_args.sample_steps)
    print(stability_dict)

    if rdkit_metrics is not None:
        rdkit_metrics = rdkit_metrics[0]
        print("Validity %.4f, Uniqueness: %.4f, Novelty: %.4f" % (rdkit_metrics[0], rdkit_metrics[1], rdkit_metrics[2]))
    else:
        print("Install rdkit roolkit to obtain Validity, Uniqueness, Novelty")

    # In GEOM-Drugs the validation partition is named 'val', not 'valid'.
    if args.dataset == 'geom':
        val_name = 'val'
        num_passes = 1
    else:
        val_name = 'valid'
        num_passes = 5

    # Evaluate negative log-likelihood for the validation and test partitions
    # val_nll = test(args, generative_model, nodes_dist, device, dtype,
    #                dataloaders[val_name],
    #                partition='Val')
    # print(f'Final val nll {val_nll}')
    # test_nll = test(args, generative_model, nodes_dist, device, dtype,
    #                 dataloaders['test'],
    #                 partition='Test', num_passes=num_passes)
    # print(f'Final test nll {test_nll}')

    # print(f'Overview: val nll {val_nll} test nll {test_nll}', stability_dict)
    # with open(join(eval_args.model_path, 'eval_log.txt'), 'w') as f:
    #     print(f'Overview: val nll {val_nll} test nll {test_nll}',
    #           stability_dict,
    #           file=f)


if __name__ == "__main__":
    main()
