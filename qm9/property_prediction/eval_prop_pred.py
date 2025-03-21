import sys, os
sys.path.append(os.path.abspath(os.path.join('../../')))
from qm9.property_prediction.models_property import EGNN, Naive, NumNodes
import torch
from torch import nn, optim
import argparse
from qm9.property_prediction import prop_utils
import json
from qm9 import dataset, utils
import pickle
from egnn.models import EGNN_dynamics_QM9
from os.path import join
from configs.datasets_config import get_dataset_info
from qm9.models import get_optim, get_model
from equivariant_diffusion import en_diffusion

loss_l1 = nn.L1Loss()


def normalize(x, atom_type, charge, node_mask):
    norm_values = [1,4,10]
    norm_biases = [0,0,0]
    x = x / norm_values[0]

    # Casting to float in case h still has long or int type.
    h_cat = (atom_type.float() - norm_biases[1]) / norm_values[1] * node_mask
    h_int = (charge.float() - norm_biases[2]) / norm_values[2]

    h_int = h_int * node_mask

    # Create new h dictionary.
    # h = {'categorical': h_cat, 'integer': h_int}

    return x, h_cat, h_int


def train(model:EGNN_dynamics_QM9, epoch, loader, mean, mad, property, device, partition='train', optimizer=None, lr_scheduler=None, log_interval=20, debug_break=False, property_pred=1, target_property="lumo", atom_type_pred=False):
    if property_pred:
        assert target_property is not None
        print("target_property: ", target_property)
    if partition == 'train':
        lr_scheduler.step()
    res = {'loss': 0, 'counter': 0, 'loss_arr':[]}
    for i, data in enumerate(loader):
        if partition == 'train':
            model.train()
            optimizer.zero_grad()

        else:
            model.eval()
        # print("data: ", data.keys())
        batch_size, n_nodes, _ = data['positions'].size()
        atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, torch.float32)
        charges = data['charges'].view(batch_size * n_nodes, -1).to(device, torch.float32)
        atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, torch.float32)
        edge_mask = data['edge_mask'].to(device, torch.float32)
        nodes = data['one_hot'].to(device, torch.float32)
        #charges = data['charges'].to(device, dtype).squeeze(2)
        #nodes = prop_utils.preprocess_input(one_hot, charges, args.charge_power, charge_scale, device)

        nodes = nodes.view(batch_size * n_nodes, -1)
        # nodes = torch.cat([one_hot, charges], dim=1)
        edges = prop_utils.get_adj_matrix(n_nodes, batch_size, device)
        if property_pred:
            label = data[target_property].to(device, torch.float32)
            # print("label: ", label)
        else:
            assert False, "property_pred should be True"
            label = data[property].to(device, torch.float32)
            
            
        atom_positions, nodes, charges = normalize(atom_positions, nodes, charges, atom_mask)
            
        xh = torch.cat([atom_positions, nodes, charges], dim=1)
        xh = xh.view(batch_size, n_nodes, -1)
        # print("xh shape: ", xh.shape)
        # print("xh[0]: ", xh[0])
        # print("label shape: ", label.shape)
        '''
        print("Positions mean")
        print(torch.mean(torch.abs(atom_positions)))
        print("Positions max")
        print(torch.max(atom_positions))
        print("Positions min")
        print(torch.min(atom_positions))


        print("\nOne hot mean")
        print(torch.mean(torch.abs(nodes)))
        print("one_hot max")
        print(torch.max(nodes))
        print("one_hot min")
        print(torch.min(nodes))


        print("\nLabel mean")
        print(torch.mean(torch.abs(label)))
        print("label max")
        print(torch.max(label))
        print("label min")
        print(torch.min(label))
        '''

        pred = model.DGAP_prop_pred(t=torch.zeros(batch_size, 1).to(device), xh=xh, edge_mask=edge_mask, node_mask=atom_mask, atom_type_pred=atom_type_pred)

        # print(f'Model device: {next(model.parameters()).device}')
        # print(f'Inputs device: {nodes.device} {atom_positions.device}')
        # print(f'Targets device: {pred.device}')
        # print("\nPred mean")
        # print(torch.mean(torch.abs(pred)))
        # print("Label shape: ", label.shape, "pred shape: ", pred.shape)
        # print("\nLabel mean")
        # print(torch.mean(torch.abs(label)))
        # print("label max")
        # print(torch.max(label))
        # print("label min")
        # print(torch.min(label))
        # print("Pred max")
        # print(torch.max(pred))
        # print("Pred min")
        # print(torch.min(pred))

        if partition == 'train':
            loss = loss_l1(pred, (label - mean) / mad)
            loss.backward()
            optimizer.step()
        else:
            if property_pred: #the output of property_pred didn't be reparametered
                # print("pred: ", pred)
                # print("label: ", label)
                loss = loss_l1(pred, label)
                # loss = loss_l1(pred, (label - mean) / mad)
                # loss = loss_l1(mad * pred + mean, label)
                # print("loss: ", loss)
            else:
                loss = loss_l1(pred, label)
                # loss = loss_l1(mad * pred + mean, label)

        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())

        prefix = ""
        if partition != 'train':
            prefix = ">> %s \t" % partition

        if i % log_interval == 0:
            print(prefix + "Epoch %d \t Iteration %d \t loss %.4f" % (epoch, i, sum(res['loss_arr'][-10:])/len(res['loss_arr'][-10:])))
        if debug_break:
            break
    print("res[loss]: ", res['loss'], "res[counter]: ", res['counter'])
    return res['loss'] / res['counter']


def test(args_gen, model, epoch, loader, mean, mad, property, device, log_interval, debug_break=False):
    # if not "property_pred" in args_gen:
    #     args_gen.property_pred = False
    #     print("[ERROR]")
    # if not "target_property" in args_gen:
    #     args_gen.target_property = None
    print("property_pred: ", args_gen.property_pred)
    print("target_property: ", args_gen.target_property)
    return train(model, epoch, loader, mean, mad, property, device, partition='test', log_interval=log_interval, debug_break=debug_break, property_pred=args_gen.property_pred, target_property=args_gen.target_property)

def get_args_gen(dir_path):
    with open(join(dir_path, 'args.pickle'), 'rb') as f:
        args_gen = pickle.load(f)
    # print("args_gen.dataset: ", args_gen.dataset)
    if args_gen.property_pred == 0:
        assert args_gen.dataset == 'qm9_second_half'

    # Add missing args!
    if not hasattr(args_gen, 'normalization_factor'):
        args_gen.normalization_factor = 1
    if not hasattr(args_gen, 'aggregation_method'):
        args_gen.aggregation_method = 'sum'
    return args_gen

# def get_model(args):
#     if args.model == 'DGAP':
#         dynamics_in_node_nf = 7
#         condition_decoupling = False
#         uni_diffusion = False
#         use_basis = False
#         decoupling = False
#         pretrain = False
#         finetune = False
#         model = EGNN_dynamics_QM9(
#         in_node_nf=dynamics_in_node_nf, context_node_nf=args.context_node_nf,
#         n_dims=3, device=device, hidden_nf=args.nf,
#         act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
#         attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
#         inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
#         normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method, condition_decoupling=condition_decoupling, uni_diffusion=uni_diffusion, use_basis=use_basis, decoupling=decoupling, pretraining=pretrain, finetune=finetune, 
#         property_pred=args.property_pred, prediction_threshold_t=args.prediction_threshold_t, target_property=args.target_property)
#         #change the in_node_nf to 22 to adapt the one_hot dimension
#         # model = EGNN(in_node_nf=22, in_edge_nf=0, hidden_nf=args.nf, device=args.device, n_layers=args.n_layers,
#         #              coords_weight=1.0,
#         #              attention=args.attention, node_attr=args.node_attr)


#     return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='QM9 Example')
    # parser.add_argument('--exp_name', type=str, default='debug', metavar='N',
                        # help='experiment_name')
    parser.add_argument('--batch_size', type=int, default=96, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test_interval', type=int, default=1, metavar='N',
                        help='how many epochs to wait before logging test')
    parser.add_argument('--outf', type=str, default='outputs', metavar='N',
                        help='folder to output vae')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                        help='learning rate')
    parser.add_argument('--nf', type=int, default=128, metavar='N',
                        help='learning rate')
    parser.add_argument('--attention', type=int, default=1, metavar='N',
                        help='attention in the ae model')
    parser.add_argument('--n_layers', type=int, default=7, metavar='N',
                        help='number of layers for the autoencoder')
    parser.add_argument('--property', type=str, default='alpha', metavar='N',
                        help='label to predict: alpha | gap | homo | lumo | mu | Cv | G | H | r2 | U | U0 | zpve')
    parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                        help='number of workers for the dataloader')
    parser.add_argument('--filter_n_atoms', type=int, default=None,
                        help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
    parser.add_argument('--charge_power', type=int, default=2, metavar='N',
                        help='maximum power to take into one-hot features')
    parser.add_argument('--dataset', type=str, default="qm9_first_half", metavar='N',
                        help='qm9_first_half')
    parser.add_argument('--datadir', type=str, default="../../qm9/temp", metavar='N',
                        help='qm9_first_half')
    parser.add_argument('--remove_h', action='store_true')
    parser.add_argument('--include_charges', type=eval, default=True, help='include atom charge or not')
    parser.add_argument('--node_attr', type=int, default=0, metavar='N',
                        help='node_attr or not')
    parser.add_argument('--weight_decay', type=float, default=1e-16, metavar='N',
                        help='weight decay')
    parser.add_argument('--save_path', type=float, default=1e-16, metavar='N',
                        help='weight decay')
    parser.add_argument('--model_name', type=str, default='numnodes', metavar='N',
                        help='egnn | naive | numnodes')
    parser.add_argument('--save_model', type=eval, default=True)
    parser.add_argument('--finetune', type=int, default=0,)
    parser.add_argument('--generators_path', type=str, default='../../outputs/edm_qm9_DGAP_resume', metavar='N', help='path to generators')
    parser.add_argument('--model_path', type=str, default='generative_model_ema_940.npy')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    dtype = torch.float32
    args.device = device
    print(args)

    res = {'epochs': [], 'losess': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}

    # prop_utils.makedir(args.outf)
    # prop_utils.makedir(args.outf + "/" + args.exp_name)

    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
    # args.dataset = "qm9_second_half"
    dataloaders_aux, _ = dataset.retrieve_dataloaders(args)
    # dataloaders["test"] = dataloaders_aux["test"]

    # compute mean and mean absolute deviation
    # property_norms = utils.compute_mean_mad_from_dataloader(dataloaders['valid'], [args.property])
    property_norms = utils.compute_mean_mad_from_dataloader(dataloaders['train'], [args.property])
    mean, mad = property_norms[args.property]['mean'], property_norms[args.property]['mad']

    args_gen = get_args_gen(args.generators_path)
    
    args_gen.branch_layers_num = 8
    print(f"\nargs_gen:", args_gen)
    dataset_info = get_dataset_info(args_gen.dataset, args_gen.remove_h, finetune=args_gen.finetune)
    # Create EGNN flow
    model, nodes_dist, prop_dist = get_model(args_gen, device, dataset_info, dataloaders['train'], uni_diffusion=args_gen.uni_diffusion, use_basis=args_gen.use_basis, decoupling=args_gen.decoupling, finetune=args_gen.finetune)
    # print(model)
    print("GPU available count: ", torch.cuda.device_count())
    # Initialize dataparallel if enabled and possible.
    # if torch.cuda.device_count() > 1:
    #     print(f'Training using {torch.cuda.device_count()} GPUs')
    #     model_dp = torch.nn.DataParallel(model)#model.cpu()
    #     model_dp = model_dp.cuda()
    # else:
    #     model_dp = model
    
    # print(model)
    #从generators_path中加载模型参数
    model_path = join(args.generators_path, args.model_path)
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path))
    model = model.dynamics.to(device)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    epoch = 1
    print("mean: ", mean, "mad: ", mad)
    
    val_loss = train(model, epoch, dataloaders['valid'], mean, mad, args.property, device,partition='valid', optimizer=optimizer, lr_scheduler=lr_scheduler, log_interval=args.log_interval, atom_type_pred=args_gen.atom_type_pred, target_property=args.property)
    # test_loss = test(args_gen, model, epoch, dataloaders['test'], mean, mad, args.property, device,log_interval=args.log_interval)
    test_loss = train(model, epoch, dataloaders['test'], mean, mad, args.property, device,partition='test', optimizer=optimizer, lr_scheduler=lr_scheduler, log_interval=args.log_interval, atom_type_pred=args_gen.atom_type_pred, target_property=args.property)
    res['epochs'].append(epoch)
    res['losess'].append(test_loss)
    if val_loss < res['best_val']:
        res['best_val'] = val_loss
        res['best_test'] = test_loss
        res['best_epoch'] = epoch
            
    print("Val loss: %.4f \t test loss: %.4f \t epoch %d" % (val_loss, test_loss, epoch))
    print("Best: val loss: %.4f \t test loss: %.4f \t epoch %d" % (res['best_val'], res['best_test'], res['best_epoch']))

        # json_object = json.dumps(res, indent=4)
        # with open(args.outf + "/" + args.exp_name + "/losess.json", "w") as outfile:
        #     outfile.write(json_object)

