from rdkit import Chem
import torch
import rdkit
from rdkit import RDLogger
from inspect import getmembers, isfunction
from rdkit.Chem import Descriptors
import time
from collections import OrderedDict

import numpy as np


with open('./property_name.txt', 'r') as f:
    names = [n.strip() for n in f.readlines()][:53]

descriptor_dict = OrderedDict()
for n in names:
    if n == 'QED':
        descriptor_dict[n] = lambda x: Chem.QED.qed(x)
    else:
        descriptor_dict[n] = getattr(Descriptors, n)

def calculate_property(mol, mean, std):
    RDLogger.DisableLog('rdApp.*')
    # mol = Chem.MolFromSmiles(files)
    output = []
    for i, descriptor in enumerate(descriptor_dict):
        # print(descriptor)
        output.append(descriptor_dict[descriptor](mol))
    output = list((np.array(output) - mean) / std)
    return torch.tensor(output, dtype=torch.float)


# statistic the qm9


if __name__=="__main__":
    from tqdm import tqdm
    qm9_mols = np.load('/data/protein/SKData/Pretraining-Denoising/fix_qm9_mols.npy', allow_pickle=True)
    prop_lst = []
    error_cnt = 0
    for i, mol in tqdm(enumerate(qm9_mols)):
        try:
            props = calculate_property(mol, 0, 1)
        except Exception as e:
            print('error', e)
            error_cnt += 1
            props = torch.zeros(53, dtype=torch.float)
        prop_lst.append(props)
    
    print('error_cnt',error_cnt)
    prop_lst = torch.stack(prop_lst)
    # save the prob_lst 
    torch.save(prop_lst, 'qm9_basic_properties.pt')
    mean = prop_lst.mean(dim=0)
    std = prop_lst.std(dim=0)
    print(mean)
    print(std)
    