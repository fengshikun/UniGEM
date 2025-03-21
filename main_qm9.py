# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import copy
import utils
import argparse
import wandb
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset
from qm9.models import get_optim, get_model
from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import torch
import time
import pickle
from qm9.utils import prepare_context, compute_mean_mad
from train_test import train_epoch, test, analyze_and_save, evaluate_properties

parser = argparse.ArgumentParser(description='E3Diffusion')
parser.add_argument('--exp_name', type=str, default='debug_10')
parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics |gnn_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')

# Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
parser.add_argument('--diffusion_steps', type=int, default=500)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                    )
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')

parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--sample_batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--brute_force', type=eval, default=False,
                    help='True | False')
parser.add_argument('--actnorm', type=eval, default=True,
                    help='True | False')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
# EGNN args -->
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of layers')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=128,
                    help='number of layers')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='diff/(|diff| + norm_constant)')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='whether using or not the sin embedding')
# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='qm9',
                    help='qm9 | qm9_second_half (train only on the last 50K samples of the training dataset)')
parser.add_argument('--datadir', type=str, default='qm9/temp',
                    help='qm9 directory')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=1)
parser.add_argument('--wandb_usr', type=str)
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--save_model', type=eval, default=True,
                    help='save model')
parser.add_argument('--generate_epochs', type=int, default=1,
                    help='save model')
parser.add_argument('--num_workers', type=int, default=4, help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=10)
parser.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='arguments : homo | lumo | alpha | gap | mu | Cv' )
parser.add_argument('--resume', type=str, default=None,
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=500,
                    help='Number of samples to compute the stability')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 1],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=True,
                    help='include atom charge or not')
parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                    help="Can be used to visualize multiple times per epoch")
parser.add_argument('--normalization_factor', type=float, default=1,
                    help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean"')

parser.add_argument('--condition_decoupling', type=eval, default=False,help='decouple the conditioning from the model')

parser.add_argument('--uni_diffusion', type=int, default=0,
                    help='whether use uni diffusion of the diffusion steps')

parser.add_argument('--use_basis', type=int, default=0,
                    help='whether use basis of the model')


parser.add_argument('--evaluate_condition_generation', type=int, default=0,)
parser.add_argument('--decoupling', type=int, default=0,)
parser.add_argument('--finetune', type=int, default=0,)
parser.add_argument('--expand_diff', type=int, default=0,
                    help='whether expand the diffusion steps')


parser.add_argument('--pretrained_model', type=str, default='',)


parser.add_argument('--denoise_pretrain', type=int, default=0, help='pretrain the model only using the denoise')

parser.add_argument("--property_pred", type=int, default=0, help='whether predict properties')
parser.add_argument("--prediction_threshold_t", type=int, default=10, help='threshold for adding the loss of  property prediction')
parser.add_argument("--target_property", type=str, help='arguments : homo | lumo | alpha | gap | mu | Cv')
parser.add_argument("--use_prop_pred", type=int, default=1, help='whether use property prediction')

parser.add_argument("--freeze_gradient", type=int, default=0, help='freeze gradient for the molecular property prediction, if set true, the gradient of molecular prediction do not inflence the generation backbone.')

parser.add_argument("--basic_prob", type=int, default=0, help='whether use basic property')
parser.add_argument("--unnormal_time_step", type=int, default=0, help='using abnormal time step')
parser.add_argument("--only_noisy_node", type=int, default=0, help='only noisy node')
parser.add_argument("--half_noisy_node", type=int, default=0, help='half 0-10, half 0-1000')
parser.add_argument("--sep_noisy_node", type=int, default=0, help='half 0-10, half 10-1000')
parser.add_argument("--atom_type_pred", type=int, default=0, help='atom type prediction under the DGAP setting')
parser.add_argument("--branch_layers_num", type=int, default=0, help="branch layer number of the second branch")

parser.add_argument("--bfn_schedule", type=int, default=0, help="whether use the bfn schedule")
parser.add_argument("--sample_steps", type=int, default=1000, help="bfn sample steps")
parser.add_argument("--use_get", type=int, default=0, help="use GET network for bond prediction")
parser.add_argument("--bond_pred", type=int, default=0, help="bond prediction")

# bfn str
parser.add_argument("--bfn_str", type=int, default=0, help="using str schedule in the bfn")
parser.add_argument("--str_loss_type", type=str, default="denoise_loss", help="loss type for the str schedule")
# parser.add_argument("--str_sigma", type=float, default=0.01, help="sigma for the str schedule")
parser.add_argument("--str_sigma_x", type=float, default=0.01, help="sigma for the str schedule for coordinate")
parser.add_argument("--str_sigma_h", type=float, default=0.01, help="sigma for the str schedule for charge")
parser.add_argument("--str_schedule_norm", type=float, default=0, help="whether use input nprmal scale for the str schedule")
parser.add_argument("--temp_index", type=int, default=0, help="temp index for the str schedule")




args = parser.parse_args()
# when we use property_pred, we need to use torchmdnet for noise prediction
if args.property_pred:
    # assert args.model == 'DGAP'
    assert args.uni_diffusion == 0
    print("assert pass")

dataset_info = get_dataset_info(args.dataset, args.remove_h, finetune=args.finetune)
print("dataset_info: ", dataset_info)

atom_encoder = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']

# args, unparsed_args = parser.parse_known_args()
args.wandb_usr = utils.get_wandb_username(args.wandb_usr)

args.cuda = not args.no_cuda and torch.cuda.is_available()
print("args.no_cuda ", args.no_cuda, " torch.cuda.is_available() ", torch.cuda.is_available())
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

if args.resume is not None:
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    wandb_usr = args.wandb_usr
    normalization_factor = args.normalization_factor
    aggregation_method = args.aggregation_method
    n_epochs = args.n_epochs
    
    

    with open(join(args.resume, 'args.pickle'), 'rb') as f:
        new_args = pickle.load(f)
        
    for key, value in vars(args).items():
        if key not in new_args:
            setattr(new_args, key, value)
    args = new_args

    args.resume = resume
    args.break_train_epoch = False

    args.exp_name = exp_name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr
    args.n_epochs = n_epochs

    # Careful with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = normalization_factor
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = aggregation_method

    print(args)

utils.create_folders(args)
# print(args)


# Wandb config
if args.no_wandb:
    mode = 'disabled'
else:
    mode = 'online' if args.online else 'offline'
kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'e3_diffusion', 'config': args,
          'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
import os
os.environ["WANDB_SERVICE_WAIT"] = "60"  # 设置等待时间为60秒
wandb.init(**kwargs)
wandb.save('*.txt')

# Retrieve QM9 dataloaders
dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

data_dummy = next(iter(dataloaders['train']))


if len(args.conditioning) > 0:
    print(f'Conditioning on {args.conditioning}')
    property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
    print("property_norms", property_norms)
    context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
    context_node_nf = context_dummy.size(2)
elif args.property_pred:
    context_node_nf = 0
    property_norms = compute_mean_mad(dataloaders, [args.target_property], args.dataset)
else:
    context_node_nf = 0
    property_norms = None

args.context_node_nf = context_node_nf


# Create EGNN flow
model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'], uni_diffusion=args.uni_diffusion, use_basis=args.use_basis, decoupling=args.decoupling, finetune=args.finetune)
if prop_dist is not None:
    prop_dist.set_normalizer(property_norms) 
model = model.to(device)
optim = get_optim(args, model)
# print(model)

gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # Add large value that will be flushed.


def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def main():
    print("args.resume: ", args.resume)
    if args.resume is not None:
        assert args.start_epoch > 0
        # flow_state_dict = torch.load(join(args.resume, 'flow.npy'))
        flow_state_dict = torch.load(join(args.resume, f'generative_model_ema_{args.start_epoch}.npy'))
        # optim_state_dict = torch.load(join(args.resume, 'optim.npy'))
        optim_state_dict = torch.load(join(args.resume, f'optim_{args.start_epoch}.npy'))
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)
        print("resume from: ", join(args.resume, f'generative_model_ema_{args.start_epoch}.npy'))
    
    if args.pretrained_model:
        print("device: ", device)
        state_dict = torch.load(args.pretrained_model, map_location=device)
        
        current_model_dict = model.state_dict()
        new_state_dict = {}
        
        for k,v in state_dict.items():
            if k in current_model_dict:
                if v.size() == current_model_dict[k].size():
                    new_state_dict[k] = v
                else:
                    print('warning size not match: ', k, v.size(), current_model_dict[k].size())
            else:
                print(f"unexpected key {k} not in current model")
        # new_state_dict={k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), state_dict.values())}

        
        miss_key, unexcept_key  =  model.load_state_dict(new_state_dict, strict=False)
        print(f"load from {args.pretrained_model}, miss_key: {miss_key}")

    # Initialize dataparallel if enabled and possible.
    if args.dp and torch.cuda.device_count() > 1:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = flow_utils.EMA(args.ema_decay)

        if args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp

    best_nll_val = 1e8
    best_nll_test = 1e8
    for epoch in range(args.start_epoch, args.n_epochs):
        start_epoch = time.time()
        
        
        # for test
        # analyze_and_save(args=args, epoch=epoch, model_sample=model_ema, nodes_dist=nodes_dist,
        #                          dataset_info=dataset_info, device=device,
        #                          prop_dist=prop_dist, n_samples=args.n_stability_samples, evaluate_condition_generation=args.evaluate_condition_generation, batch_size=10)
        
        # nll_val = test(args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=model_ema_dp,
        #                    partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist,
        #                    property_norms=property_norms, uni_diffusion=args.uni_diffusion)
        # for test
        # evaluate_properties(args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=model_ema_dp,partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist,property_norms=property_norms, wandb=wandb)
        
        train_epoch(args=args, loader=dataloaders['train'], epoch=epoch, model=model, model_dp=model_dp,
                    model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
                    nodes_dist=nodes_dist, dataset_info=dataset_info,
                    gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist, uni_diffusion=args.uni_diffusion)
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")

        if epoch % args.test_epochs == 0:
            
            if args.uni_diffusion:
                # evaluate properties
                evaluate_properties(args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=model_ema_dp,partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist, property_norms=property_norms, wandb=wandb)
            
            if isinstance(model, en_diffusion.EnVariationalDiffusion):
                wandb.log(model.log_info(), commit=True)

            if not args.break_train_epoch:
                analyze_and_save(args=args, epoch=epoch, model_sample=model_ema, nodes_dist=nodes_dist,
                                 dataset_info=dataset_info, device=device,
                                 prop_dist=prop_dist, n_samples=args.n_stability_samples, evaluate_condition_generation=args.evaluate_condition_generation)
            nll_val = test(args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=model_ema_dp,
                           partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist,
                           property_norms=property_norms, uni_diffusion=args.uni_diffusion)
            nll_test = test(args=args, loader=dataloaders['test'], epoch=epoch, eval_model=model_ema_dp,
                            partition='Test', device=device, dtype=dtype,
                            nodes_dist=nodes_dist, property_norms=property_norms, uni_diffusion=args.uni_diffusion)

            if nll_val < best_nll_val:
                best_nll_val = nll_val
                best_nll_test = nll_test
                if args.save_model:
                    args.current_epoch = epoch + 1
                    utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
                    utils.save_model(model, 'outputs/%s/generative_model.npy' % args.exp_name)
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
                    with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(args, f)

            if args.save_model:
                utils.save_model(optim, 'outputs/%s/optim_%d.npy' % (args.exp_name, epoch))
                utils.save_model(model, 'outputs/%s/generative_model_%d.npy' % (args.exp_name, epoch))
                if args.ema_decay > 0:
                    utils.save_model(model_ema, 'outputs/%s/generative_model_ema_%d.npy' % (args.exp_name, epoch))
                with open('outputs/%s/args_%d.pickle' % (args.exp_name, epoch), 'wb') as f:
                    pickle.dump(args, f)
            print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
            print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_nll_val, best_nll_test))
            wandb.log({"Val loss ": nll_val}, commit=True)
            wandb.log({"Test loss ": nll_test}, commit=True)
            wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)


if __name__ == "__main__":
    main()
