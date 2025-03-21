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


