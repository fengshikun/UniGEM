import torch
from torch.utils.data import Dataset

import os
from itertools import islice
from math import inf
import numpy as np
from rdkit import Chem
from tqdm import tqdm
import pickle

import logging

class ProcessedDataset(Dataset):
    """
    Data structure for a pre-processed cormorant dataset.  Extends PyTorch Dataset.

    Parameters
    ----------
    data : dict
        Dictionary of arrays containing molecular properties.
    included_species : tensor of scalars, optional
        Atomic species to include in ?????.  If None, uses all species.
    num_pts : int, optional
        Desired number of points to include in the dataset.
        Default value, -1, uses all of the datapoints.
    normalize : bool, optional
        ????? IS THIS USED?
    shuffle : bool, optional
        If true, shuffle the points in the dataset.
    subtract_thermo : bool, optional
        If True, subtracts the thermochemical energy of the atoms from each molecule in GDB9.
        Does nothing for other datasets.

    """
    def __init__(self, data, included_species=None, num_pts=-1, normalize=True, shuffle=True, subtract_thermo=True, basic_prob=False, bond_pred=False):

        self.data = data

        if num_pts < 0:
            self.num_pts = len(data['charges'])
        else:
            if num_pts > len(data['charges']):
                logging.warning('Desired number of points ({}) is greater than the number of data points ({}) available in the dataset!'.format(num_pts, len(data['charges'])))
                self.num_pts = len(data['charges'])
            else:
                self.num_pts = num_pts

        # If included species is not specified
        if included_species is None:
            included_species = torch.unique(self.data['charges'], sorted=True)
            if included_species[0] == 0:
                included_species = included_species[1:]

        if subtract_thermo:
            thermo_targets = [key.split('_')[0] for key in data.keys() if key.endswith('_thermo')]
            if len(thermo_targets) == 0:
                logging.warning('No thermochemical targets included! Try reprocessing dataset with --force-download!')
            else:
                logging.info('Removing thermochemical energy from targets {}'.format(' '.join(thermo_targets)))
            for key in thermo_targets:
                data[key] -= data[key + '_thermo'].to(data[key].dtype)

        self.included_species = included_species

        self.data['one_hot'] = self.data['charges'].unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)

        self.num_species = len(included_species)
        self.max_charge = max(included_species)

        self.parameters = {'num_species': self.num_species, 'max_charge': self.max_charge}
        
        self.basic_prob = basic_prob
        
        self.bond_pred = bond_pred
        
        if self.bond_pred:
            self.qm9_mols_lst = Chem.SDMolSupplier('/mnt/nfs-ssd/data/fengshikun/DenoisingData/qm9/raw/gdb9.sdf', removeHs=False, sanitize=False)
        
        if self.basic_prob:
            self.qm9_mols = np.load('fix_qm9_mols.npy', allow_pickle=True)

            self.basic_probs = torch.load('qm9_basic_properties.pt')
            self.basic_probs_mean = self.basic_probs.mean(dim=0)
            self.basic_probs_std = self.basic_probs.std(dim=0)
            # qm9_root = '/data/protein/SKData/DenoisingData/qm9'
            # raw_sdf_file = os.path.join(qm9_root, 'raw/gdb9.sdf')
            # suppl = Chem.SDMolSupplier(raw_sdf_file, removeHs=False,
            #                                 sanitize=False)
            # with open(os.path.join(qm9_root, 'raw/uncharacterized.txt'), 'r') as f:
            #     skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]


            # mol_list = []
            self.cnt_map_idx_dict = {}
            
            
            
            # save self.cnt_map_idx_dict to a file
            # with open('cnt_map_idx_dict.pickle', 'wb') as handle:
            #     pickle.dump(self.cnt_map_idx_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            with open('cnt_map_idx_dict.pickle', 'rb') as handle:
                self.cnt_map_idx_dict = pickle.load(handle)
        
        # cnt = 0
        # for i, mol in enumerate(tqdm(suppl)):
        #     if i in skip:
        #         continue
        #     self.cnt_map_idx_dict[i] = cnt
        #     cnt += 1
        #     mol_list.append(mol)

        # Get a dictionary of statistics for all properties that are one-dimensional tensors.
        self.calc_stats()

        if shuffle:
            self.perm = torch.randperm(len(data['charges']))[:self.num_pts]
        else:
            self.perm = None
        
        self.basic_prob = basic_prob

    def calc_stats(self):
        self.stats = {key: (val.mean(), val.std()) for key, val in self.data.items() if type(val) is torch.Tensor and val.dim() == 1 and val.is_floating_point()}

    def convert_units(self, units_dict):
        for key in self.data.keys():
            if key in units_dict:
                self.data[key] *= units_dict[key]

        self.calc_stats()

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        if self.perm is not None:
            idx = self.perm[idx]
        
        if self.bond_pred:
            mol_idx = self.data['index'][idx].item() - 1
            mol_obj = self.qm9_mols_lst[mol_idx]
            if mol_obj is None:
                print('mol_obj is None')
                print(mol_idx)
            atom_num = mol_obj.GetNumAtoms()
            atom_num2 = self.data['num_atoms'][idx]
            if atom_num != atom_num2:
                print('atom_num != atom_num2')
                print(atom_num, atom_num2)
            assert atom_num == atom_num2
            
        
        if self.basic_prob:
            mol_idx = self.data['index'][idx].item() - 1
            mol_idx = self.cnt_map_idx_dict[mol_idx]
            mol_obj = self.qm9_mols[mol_idx]
            
            basic_prob = self.basic_probs[mol_idx]
            atom_num = mol_obj.GetNumAtoms()
            atom_num2 = self.data['num_atoms'][idx]
            # assert atom_num == atom_num2
        
        data_tmp = {key: val[idx] for key, val in self.data.items()}
        
        if self.basic_prob:
            data_tmp['basic_prob'] = (basic_prob - self.basic_probs_mean) / self.basic_probs_std
        
        if self.bond_pred:
            adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol_obj, useBO=True))
            edge_index = adj.nonzero().contiguous().T
            bond_types = adj[edge_index[0], edge_index[1]]
            bond_types[bond_types == 1.5] = 4
            edge_attr = bond_types.long()
            data_tmp['edge_index'] = edge_index
            data_tmp['edge_attr'] = edge_attr.reshape(1, -1)
        
        return data_tmp
        # return {key: val[idx] for key, val in self.data.items()}
