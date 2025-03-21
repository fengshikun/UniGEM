import wandb
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
import numpy as np
import qm9.visualizer as vis
from qm9.analyze import analyze_stability_for_molecules
from qm9.sampling import sample_chain, sample, sample_sweep_conditional, charge_decode
import utils
import qm9.utils as qm9utils
from qm9 import losses
import time
import torch
import os


def only_sample_chains(args, loader, epoch, model, model_dp, model_ema, ema, device, dtype, property_norms, optim,
                nodes_dist, gradnorm_queue, dataset_info, prop_dist, uni_diffusion=False):
    
    
    # debug
    i = 0
    start = time.time()
    save_and_sample_chain(model_ema, args, device, dataset_info, prop_dist, epoch=epoch,
                                  batch_id=str(i))
    sample_different_sizes_and_save(model_ema, nodes_dist, args, device, dataset_info,
                                    prop_dist, epoch=epoch, batch_id=str(i))
    print(f'Sampling took {time.time() - start:.2f} seconds')

    vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_{i}", dataset_info=dataset_info, wandb=wandb)
    vis.visualize_chain(f"outputs/{args.exp_name}/epoch_{epoch}_{i}/chain/", dataset_info, wandb=wandb)

def train_epoch(args, loader, epoch, model, model_dp, model_ema, ema, device, dtype, property_norms, optim,
                nodes_dist, gradnorm_queue, dataset_info, prop_dist, uni_diffusion=False):
    model_dp.train()
    model.train()
    nll_epoch = []
    n_iterations = len(loader)
    mask_indicator = False
    
    if args.denoise_pretrain:
        mask_indicator = 2
    
    
    
    for i, data in enumerate(loader):
        # if i >= 3:
        #     break
        x = data['positions'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
        
        if args.bond_pred:
            edge_index = data['edge_index'].to(device, dtype)
            edge_attr = data['edge_attr'].to(device, dtype)
            bond_info = {'edge_index': edge_index, 'edge_attr': edge_attr}
        else:
            bond_info = None

        x = remove_mean_with_mask(x, node_mask) # erase mean value

        if args.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * args.augment_noise

        x = remove_mean_with_mask(x, node_mask)
        if args.data_augmentation:
            x = utils.random_rotation(x).detach()

        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        h = {'categorical': one_hot, 'integer': charges}

        if len(args.conditioning) > 0 or ('property' in data and args.uni_diffusion):
            if 'property' in data:
                context = data['property']
                # convert the shape of context 32 x 53 to 32 x 1 x 53
                context = context.unsqueeze(1)
                # repeat the context 43 times along axis 1, to match the shape 32 x 43 x 53
                context = context.repeat(1, x.shape[1], 1).to(device, dtype)
                context = context * node_mask
            else:
                context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
                assert_correctly_masked(context, node_mask)
        else:
            context = None

        optim.zero_grad()

        # transform batch through flow
        if uni_diffusion:
            # print("x shape", x.shape)
            nll, reg_term, mean_abs_z, loss_dict = losses.compute_loss_and_nll(args, model_dp, nodes_dist,
                                                                x, h, node_mask, edge_mask, context, uni_diffusion=uni_diffusion, mask_indicator=mask_indicator)
            
            if args.denoise_pretrain:
                mask_indicator = 2
            else:
                mask_indicator = not mask_indicator
            # wandb log error and error2
            if 'denoise_error' in loss_dict:
                wandb.log({"denoise_x": loss_dict['error'].mean().item(), "denoise_error": loss_dict['denoise_error'].mean().item()}, commit=True)
            else:
                wandb.log({"denoise_x": loss_dict['error'].mean().item(), "denoise_y": loss_dict['error2'].mean().item()}, commit=True)
            
        else:
            if args.target_property in data:
                property_label = data[args.target_property].to(device, dtype)
                if property_norms is not None:
                    property_label = (property_label - property_norms[args.target_property]['mean']) / property_norms[args.target_property]['mad']
            else:
                property_label = None
                
            nll, reg_term, mean_abs_z, loss_dict = losses.compute_loss_and_nll(args, model_dp, nodes_dist,
                                                                x, h, node_mask, edge_mask, context,
                                                                 property_label=property_label, bond_info=bond_info)
            
            if 'error' in loss_dict:
                wandb.log({"denoise_x": loss_dict['error'].mean().item()}, commit=True)
            if 'pred_loss' in loss_dict:
                if isinstance(loss_dict['pred_loss'], torch.Tensor):
                    wandb.log({"pred_loss": loss_dict['pred_loss'].mean().item(), "pred_rate": loss_dict['pred_rate'].mean().item()}, commit=True)
            if 'atom_type_loss' in loss_dict:
                wandb.log({"atom_type_loss": loss_dict['atom_type_loss'].mean().item()}, commit=True)
            
            if 'posloss' in loss_dict:
                wandb.log({"posloss": loss_dict['posloss'].mean().item()}, commit=True)
            if 'charge_loss' in loss_dict:
                wandb.log({"charge_loss": loss_dict['charge_loss'].mean().item()}, commit=True)
            
            if 'bond_loss' in loss_dict:
                wandb.log({"bond_loss": loss_dict['bond_loss'].mean().item()}, commit=True)
                nll += loss_dict['bond_loss'].mean()
        # standard nll from forward KL
        loss = nll + args.ode_regularization * reg_term
        # loss.backward()

        try:
            loss.backward()

            if args.clip_grad:
                grad_norm = utils.gradient_clipping(model, gradnorm_queue)
            else:
                grad_norm = 0.

            optim.step()
        except Exception as e:
            grad_norm = 0.
            print('Error in backward pass(may occure loss zero), skipping batch')

        optim.step()

        # Update EMA if enabled.
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model)

        if i % args.n_report_steps == 0:
            if uni_diffusion:
                if 'error2' in loss_dict:
                    print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                      f"RegTerm: {reg_term.item():.1f}, "
                      f"GradNorm: {grad_norm:.1f}, "
                      f"denoise x: {loss_dict['error'].mean().item():.3f}",
                      f"denoise y: {loss_dict['error2'].mean().item():.3f}")
                else:
                    print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                      f"RegTerm: {reg_term.item():.1f}, "
                      f"GradNorm: {grad_norm:.1f}, "
                      f"denoise x: {loss_dict['error'].mean().item():.3f}, "
                      f"denoise only x: {loss_dict['denoise_error'].mean().item():.3f}")
            else:
                if 'error' in loss_dict:
                    print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                        f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                        f"RegTerm: {reg_term.item():.1f}, "
                        f"GradNorm: {grad_norm:.1f}, "
                        f"denoise x: {loss_dict['error'].mean().item():.3f} ", 
                        end='' if args.property_pred or args.model == "PAT" else '\n')
                else: # BFN
                    print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                        f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                        f"RegTerm: {reg_term.item():.1f}, "
                        f"posloss: {loss_dict['posloss'].mean().item():.3f}, "
                        f"charge_loss: {loss_dict['charge_loss'].mean().item():.3f}, "
                        f"GradNorm: {grad_norm:.1f}", end='' if args.property_pred or args.model == "PAT" else '\n')
                if args.bond_pred:
                    print(f", bond_loss: {loss_dict['bond_loss'].mean():.3f}", end='')
                
                if args.property_pred:
                    if not isinstance(loss_dict['pred_loss'], int):
                        print(f", pred_loss: {loss_dict['pred_loss'].mean():.3f}", end='')
                    print(f", pred_rate: {loss_dict['pred_rate'].mean():.3f}")
                if args.model == "PAT":
                    print(f', atom_type_loss: {loss_dict["atom_type_loss"].mean():.3f}', end='')
                    print(f", pred_rate: {loss_dict['pred_rate'].mean():.3f}")
        nll_epoch.append(nll.item())
        # save_and_sample_chain(model_ema, args, device, dataset_info, prop_dist, epoch=epoch,
        #                           batch_id=str(i))
        # sample_different_sizes_and_save(model_ema, nodes_dist, args, device, dataset_info,
        #                                 prop_dist, epoch=epoch)
        if (epoch % args.test_epochs == 0) and (i % args.visualize_every_batch == 0) and not (epoch == 0 and i == 0):
        # if epoch == 0: # for test
            start = time.time()
            if len(args.conditioning) > 0 and not args.uni_diffusion:
                save_and_sample_conditional(args, device, model_ema, prop_dist, dataset_info, epoch=epoch)
            save_and_sample_chain(model_ema, args, device, dataset_info, prop_dist, epoch=epoch,
                                  batch_id=str(i))
            sample_different_sizes_and_save(model_ema, nodes_dist, args, device, dataset_info,
                                            prop_dist, epoch=epoch)
            print(f'Sampling took {time.time() - start:.2f} seconds')

            vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_{i}", dataset_info=dataset_info, wandb=wandb)
            vis.visualize_chain(f"outputs/{args.exp_name}/epoch_{epoch}_{i}/chain/", dataset_info, wandb=wandb)
            if len(args.conditioning) > 0 and not args.uni_diffusion:
                vis.visualize_chain("outputs/%s/epoch_%d/conditional/" % (args.exp_name, epoch), dataset_info,
                                    wandb=wandb, mode='conditional')
        wandb.log({"Batch NLL": nll.item()}, commit=True)
        if args.break_train_epoch:
            break
    wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def evaluate_properties(args, loader, epoch, eval_model, device, dtype, property_norms, nodes_dist, partition='Test', wandb=None): # node properties evaluation
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        n_iterations = len(loader)
        
        gts = []
        preds = []
        
        for i, data in enumerate(loader):
            x = data['positions'].to(device, dtype)
            batch_size = x.size(0)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
            
            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            h = {'categorical': one_hot, 'integer': charges}
            
            if 'property' in data:
                context = data['property']
                context = context.unsqueeze(1)
                context = context.repeat(1, x.shape[1], 1).to(device, dtype)
                org_context = context * node_mask
            else:
                org_context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            assert_correctly_masked(org_context, node_mask)
            if isinstance(eval_model, torch.nn.DataParallel):
                pred_properties, batch_mae = eval_model.module.evaluate_property(x, h, org_context, node_mask, edge_mask)
            else:
                pred_properties, batch_mae = eval_model.evaluate_property(x, h, org_context, node_mask, edge_mask)
            
            preds.append(pred_properties)
            gts.append(org_context)
            
            print(f'batch mae is {batch_mae}')
            
            break # for test speed up
        
        # calculate the mean absolute error between preds and gts
        preds = torch.cat(preds, dim=0)
        gts = torch.cat(gts, dim=0)
        preds = preds[:, 0, 0]
        gts = gts[:, 0, 0]
        mae = torch.mean(torch.abs(preds - gts))
        
        if wandb is not None:
            wandb.log({'Properties Mean Absolute Error': mae.item()})
        
        print(f'Epoch {epoch}: properties Mean Absolute Error is {mae}')
        
        

def test(args, loader, epoch, eval_model, device, dtype, property_norms, nodes_dist, partition='Test', uni_diffusion=False):
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        n_iterations = len(loader)

        for i, data in enumerate(loader):
            x = data['positions'].to(device, dtype)
            batch_size = x.size(0)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
            
            if args.bond_pred:
                edge_index = data['edge_index'].to(device, dtype)
                edge_attr = data['edge_attr'].to(device, dtype)
                bond_info = {'edge_index': edge_index, 'edge_attr': edge_attr}
            else:
                bond_info = None

            if args.augment_noise > 0:
                # Add noise eps ~ N(0, augment_noise) around points.
                eps = sample_center_gravity_zero_gaussian_with_mask(x.size(),
                                                                    x.device,
                                                                    node_mask)
                x = x + eps * args.augment_noise

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            h = {'categorical': one_hot, 'integer': charges}

            if len(args.conditioning) > 0:
                context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
                assert_correctly_masked(context, node_mask)
            elif 'property' in data:
                context = data['property']
                context = context.unsqueeze(1)
                context = context.repeat(1, x.shape[1], 1).to(device, dtype)
                context = context * node_mask
            else:
                context = None

            # transform batch through flow
            if uni_diffusion:
                nll, _, _, _ = losses.compute_loss_and_nll(args, eval_model, nodes_dist, x, h,
                                                            node_mask, edge_mask, context, uni_diffusion=uni_diffusion)
            else:
                nll, _, _, _ = losses.compute_loss_and_nll(args, eval_model, nodes_dist, x, h,
                                                    node_mask, edge_mask, context, uni_diffusion=uni_diffusion
                                                    , property_label=data[args.target_property].to(device, dtype) if args.target_property in data else None, bond_info=bond_info)
            # standard nll from forward KL

            nll_epoch += nll.item() * batch_size
            n_samples += batch_size
            if i % args.n_report_steps == 0:
                print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"NLL: {nll_epoch/n_samples:.2f}")

    return nll_epoch/n_samples


def save_and_sample_chain(model, args, device, dataset_info, prop_dist,
                          epoch=0, id_from=0, batch_id=''):
    one_hot, charges, x = sample_chain(args=args, device=device, flow=model,
                                       n_tries=1, dataset_info=dataset_info, prop_dist=prop_dist)

    vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/',
                      one_hot, charges, x, dataset_info, id_from, name='chain', bfn_schedule=args.bfn_schedule)

    return one_hot, charges, x


def save_and_sample_chain_save(model, args, device, dataset_info, prop_dist,
                          epoch=0, id_from=0, batch_id=''):
    one_hot, charges, x = sample_chain(args=args, device=device, flow=model,
                                       n_tries=1, dataset_info=dataset_info, prop_dist=prop_dist)

    
    # write_dir = f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/'
    # if not os.path.exists(write_dir):
    #     os.makedirs(write_dir)
    vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/',
                      one_hot, charges, x, dataset_info, id_from, name='chain', bfn_schedule=args.bfn_schedule)

    vis.visualize_chain(f"outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/", dataset_info, wandb=None)    

    return one_hot, charges, x


def sample_different_sizes_and_save(model, nodes_dist, args, device, dataset_info, prop_dist,
                                    n_samples=5, epoch=0, batch_size=100, batch_id=''):
    batch_size = min(batch_size, n_samples)
    for counter in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        if args.bfn_schedule:
            theta_traj, segment_ids = sample(args, device, model, prop_dist=prop_dist,
                                                nodesxsample=nodesxsample, dataset_info=dataset_info)
        elif args.property_pred:
            one_hot, charges, x, node_mask, pred = sample(args, device, model, prop_dist=prop_dist,
                                                nodesxsample=nodesxsample,
                                                dataset_info=dataset_info)
        else:
            one_hot, charges, x, node_mask = sample(args, device, model, prop_dist=prop_dist,
                                                nodesxsample=nodesxsample,
                                                dataset_info=dataset_info)
        if not args.bfn_schedule:
            print(f"Generated molecule: Positions {x[:-1, :, :]}")
        
        if args.bfn_schedule:
            frame_num = len(theta_traj)
            charges = []
            one_hot = []
            x = []
            for i in range(frame_num):
                x.append(theta_traj[i][0].cpu().numpy())
                h = theta_traj[i][1].cpu()
                one_hot.append(charge_decode(h, dataset_info))
        
        vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/', one_hot, charges, x, dataset_info,
                          batch_size * counter, name='molecule', bfn_schedule=args.bfn_schedule)


# def charge_decode(charge, dataset_info, remove_h=False):
#     atomic_nb = dataset_info['atomic_nb']
#     atom_type_num = len(atomic_nb[remove_h:])
#     anchor = torch.tensor(
#         [
#             (2 * k - 1) / max(atomic_nb) - 1
#             for k in atomic_nb[remove_h :]
#         ],
#         dtype=torch.float32,
#         device=charge.device,
#     )
#     atom_type = (charge - anchor).abs().argmin(dim=-1)
#     one_hot = torch.zeros(
#         [charge.shape[0], atom_type_num], dtype=torch.float32
#     )
#     one_hot[torch.arange(charge.shape[0]), atom_type] = 1
#     return one_hot

def analyze_and_save(epoch, model_sample, nodes_dist, args, device, dataset_info, prop_dist,
                     n_samples=1000, batch_size=100, evaluate_condition_generation=False):
    print(f'Analyzing molecule stability at epoch {epoch}...')
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    if args.bfn_schedule:
        molecules = []
    else:
        molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    for i in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        if args.bfn_schedule:
            theta_traj, segment_ids = sample(args, device, model_sample, dataset_info, prop_dist,
                                                nodesxsample=nodesxsample, evaluate_condition_generation=evaluate_condition_generation)
        elif args.property_pred:
            one_hot, charges, x, node_mask, pred = sample(args, device, model_sample, dataset_info, prop_dist,
                                                nodesxsample=nodesxsample, evaluate_condition_generation=evaluate_condition_generation)
        else:
            one_hot, charges, x, node_mask = sample(args, device, model_sample, dataset_info, prop_dist,
                                                nodesxsample=nodesxsample, evaluate_condition_generation=evaluate_condition_generation)

        # processed_list.append((pos, atom_type))
        if args.bfn_schedule:
            segment_ids = segment_ids.cpu()
            x, h = theta_traj[-1] # x: N x 3
            if h.shape[-1] == 1:
                atom_types = charge_decode(h, dataset_info) # N x 1 --> N x 5
            else:
                atom_types = h #  atom type prediction: h is N x 5 logits
            bz = segment_ids.max().item()
            for i in range(bz):
                pos = x[segment_ids == i].detach().cpu()
                sub_atom_type = atom_types[segment_ids == i] # n x 5
                sub_atom_type = sub_atom_type.argmax(1).cpu()
                molecules.append((pos, sub_atom_type))
        else:
            molecules['one_hot'].append(one_hot.detach().cpu())
            molecules['x'].append(x.detach().cpu())
            molecules['node_mask'].append(node_mask.detach().cpu())

    if not args.bfn_schedule:
        molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, dataset_info, bfn_schedule=args.bfn_schedule)

    wandb.log(validity_dict)
    if rdkit_tuple is not None:
        wandb.log({'Validity': rdkit_tuple[0][0], 'Uniqueness': rdkit_tuple[0][1], 'Novelty': rdkit_tuple[0][2]})
    return validity_dict


def save_and_sample_conditional(args, device, model, prop_dist, dataset_info, epoch=0, id_from=0):
    if args.property_pred:
        one_hot, charges, x, node_mask, pred = sample_sweep_conditional(args, device, model, dataset_info, prop_dist)
    else:
        one_hot, charges, x, node_mask = sample_sweep_conditional(args, device, model, dataset_info, prop_dist)

    vis.save_xyz_file(
        'outputs/%s/epoch_%d/conditional/' % (args.exp_name, epoch), one_hot, charges, x, dataset_info,
        id_from, name='conditional', node_mask=node_mask)

    return one_hot, charges, x
