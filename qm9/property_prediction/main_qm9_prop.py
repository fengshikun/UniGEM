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
import numpy as np

loss_l1 = nn.L1Loss()
loss_l1_nr = nn.L1Loss(reduction='none')


def train(model, epoch, loader, mean, mad, property, device, partition='train', optimizer=None, lr_scheduler=None, log_interval=20, debug_break=False, property_pred=0, target_property=None, ):
    if property_pred:
        assert target_property is not None
        print("target_property: ", target_property)
    if partition == 'train':
        lr_scheduler.step()
    res = {'loss': 0, 'counter': 0, 'loss_arr':[]}
    
    if loader.dataset_info is not None:
        loss_lst = []
        stable_lst = []
    
    
    for i, data in enumerate(loader):
        if partition == 'train':
            model.train()
            optimizer.zero_grad()

        else:
            model.eval()

        batch_size, n_nodes, _ = data['positions'].size()
        atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, torch.float32)
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
        else:
            label = data[property].to(device, torch.float32)

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
        
        if 'EGNN_dynamics_QM9' in type(model).__name__: #if the model is EnVariationalDiffusion
            # id2charge = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
            idx2charge = [1, 6, 7, 8, 9]
            
            # get charges from nodes which is one hot encoding
            charges = torch.tensor([idx2charge[torch.argmax(node).item()] for node in nodes]).to(device)
            # * atom_mask
            charges = charges.unsqueeze(1) * atom_mask
            
            # TODO normalize nodes and charges
            
            # concat charges to nodes
            nodes = torch.cat([nodes, charges], dim=1)
            
            # atom type pred, the nodes convert to 1
            # nodes = torch.ones_like(nodes) * atom_mask
            
            nodes = nodes.reshape(batch_size, n_nodes, -1)
            atom_positions = atom_positions.reshape(batch_size, n_nodes, -1)
            xh = torch.cat([atom_positions, nodes], dim=2)
            # atom_mask = atom_mask.reshape(batch_size, n_nodes, -1)
            
            
            pred = model.DGAP_prop_pred(t=torch.zeros(batch_size, 1).to(device), xh=xh, edge_mask=edge_mask, node_mask=atom_mask, atom_type_pred=True)
            
            
            # pred = model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
            #          n_nodes=n_nodes, target=label)
        else:
            pred = model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
                        n_nodes=n_nodes)

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
                print("pred: ", pred)
                print("label: ", label)
                # loss = loss_l1(pred, label)
                loss = loss_l1(1.0525 * pred + 0.321, label)
                # loss = loss_l1(mad * pred + mean, label)
                print("loss: ", loss)
            else:
                if property == 'lumo' and 'EGNN_dynamics_QM9' in type(model).__name__:
                    loss = loss_l1(pred, label)
                else:
                    loss = loss_l1(mad * pred + mean, label)

        
        if 'stability' in data:
            if property == 'lumo' and 'EGNN_dynamics_QM9' in type(model).__name__:
                c_loss_lst = loss_l1_nr(pred, label)
            else:
                c_loss_lst = loss_l1_nr(mad * pred + mean, label)
            
            # c_loss_lst = loss_l1_nr(pred, label)
            c_loss_lst = c_loss_lst.cpu().detach().tolist()
            loss_lst.extend(c_loss_lst)
            stable_lst.extend(data['stability'])
        
        
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
    
    if 'stability' in data:
        loss_lst = np.array(loss_lst)
        stable_lst = np.array(stable_lst)
        # stable rate
        stable_rate = np.sum(stable_lst) / len(stable_lst)
        # average loss
        avg_loss = np.mean(loss_lst)
        # stable loss
        stable_lst = stable_lst.astype(bool)
        stable_loss = np.mean(loss_lst[stable_lst])
        # unstable loss
        unstable_loss = np.mean(loss_lst[~stable_lst])
        print(f'Average loss: {avg_loss:.4f}, stable loss: {stable_loss:.4f}, unstable loss: {unstable_loss:.4f}, stable rate: {stable_rate:.4f}')
        
        
    
    return res['loss'] / res['counter']


def test(args_gen, model, epoch, loader, mean, mad, property, device, log_interval, debug_break=False):
    if not "property_pred" in args_gen:
        args_gen.property_pred = False
    if not "target_property" in args_gen:
        args_gen.target_property = None
    return train(model, epoch, loader, mean, mad, property, device, partition='test', log_interval=log_interval, debug_break=debug_break, property_pred=args_gen.property_pred, target_property=args_gen.target_property)


def get_model(args):
    if args.model_name == 'egnn':
        model = EGNN(in_node_nf=5, in_edge_nf=0, hidden_nf=args.nf, device=args.device, n_layers=args.n_layers,
                     coords_weight=1.0,
                     attention=args.attention, node_attr=args.node_attr)
        #change the in_node_nf to 22 to adapt the one_hot dimension
        # model = EGNN(in_node_nf=22, in_edge_nf=0, hidden_nf=args.nf, device=args.device, n_layers=args.n_layers,
        #              coords_weight=1.0,
        #              attention=args.attention, node_attr=args.node_attr)
    elif args.model_name == 'naive':
        model = Naive(device=args.device)
    elif args.model_name == 'numnodes':
        model = NumNodes(device=args.device)
    else:
        raise Exception("Wrong model name %s" % args.model_name)


    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='QM9 Example')
    parser.add_argument('--exp_name', type=str, default='debug', metavar='N',
                        help='experiment_name')
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

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    dtype = torch.float32
    args.device = device
    print(args)

    res = {'epochs': [], 'losess': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}

    prop_utils.makedir(args.outf)
    prop_utils.makedir(args.outf + "/" + args.exp_name)


    if args.dataset == "qm9":
        dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
    else:
        dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
        args.dataset = "qm9_second_half"
        dataloaders_aux, _ = dataset.retrieve_dataloaders(args)
        dataloaders["test"] = dataloaders_aux["train"]

    # compute mean and mean absolute deviation
    property_norms = utils.compute_mean_mad_from_dataloader(dataloaders['valid'], [args.property])
    mean, mad = property_norms[args.property]['mean'], property_norms[args.property]['mad']

    model = get_model(args)
    print("GPU available count: ", torch.cuda.device_count())
    # Initialize dataparallel if enabled and possible.
    if torch.cuda.device_count() > 1:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        model_dp = torch.nn.DataParallel(model)#model.cpu()
        model_dp = model_dp.cuda()
    else:
        model_dp = model

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    print("epochs: ", args.epochs)

    for epoch in range(0, args.epochs):
        train(model, epoch, dataloaders['train'], mean, mad, args.property, device, partition='train', optimizer=optimizer, lr_scheduler=lr_scheduler, log_interval=args.log_interval)
        if epoch % args.test_interval == 0:
            val_loss = train(model, epoch, dataloaders['valid'], mean, mad, args.property, device, partition='valid', optimizer=optimizer, lr_scheduler=lr_scheduler, log_interval=args.log_interval)
            test_loss = test(args, model, epoch, dataloaders['test'], mean, mad, args.property, device, log_interval=args.log_interval)
            res['epochs'].append(epoch)
            res['losess'].append(test_loss)

            if val_loss < res['best_val']:
                res['best_val'] = val_loss
                res['best_test'] = test_loss
                res['best_epoch'] = epoch
                if args.save_model:
                    torch.save(model.state_dict(), args.outf + "/" + args.exp_name + "/best_checkpoint.npy")
                    with open(args.outf + "/" + args.exp_name + "/args.pickle", 'wb') as f:
                        pickle.dump(args, f)
            print("Val loss: %.4f \t test loss: %.4f \t epoch %d" % (val_loss, test_loss, epoch))
            print("Best: val loss: %.4f \t test loss: %.4f \t epoch %d" % (res['best_val'], res['best_test'], res['best_epoch']))

        json_object = json.dumps(res, indent=4)
        with open(args.outf + "/" + args.exp_name + "/losess.json", "w") as outfile:
            outfile.write(json_object)

