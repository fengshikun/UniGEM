import build_geom_dataset
from tqdm import tqdm
import torch
from rdkit import Chem
import yaml

pcq_with_h = {'name': 'pcq', 
              'atom_encoder': {'H': 0, 'He': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Mg': 8, 'Si': 9, 'P': 10, 'S': 11, 'Cl': 12, 'Ar': 13, 'Ca': 14, 'Ti': 15, 'Zn': 16, 'Ga': 17, 'Ge': 18, 'As': 19, 'Se': 20, 'Br': 21}, 
              'atomic_nb': [1, 2, 4, 5, 6, 7, 8, 9, 12, 14, 15, 16, 17, 18, 20, 22, 30, 31, 32, 33, 34, 35], 
              'atom_decoder': ['H', 'He', 'Be', 'B', 'C', 'N', 'O', 'F', 'Mg', 'Si', 'P', 'S', 'Cl', 'Ar', 'Ca', 'Ti', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br'], 
              'max_n_nodes': 53, 
              'n_nodes': {2: 77, 3: 62, 4: 162, 5: 439, 6: 721, 7: 1154, 8: 1879, 9: 2758, 10: 4419, 11: 6189, 12: 9283, 13: 12620, 14: 18275, 15: 24340, 16: 31938, 17: 40477, 18: 51301, 19: 62211, 20: 74714, 21: 87453, 22: 103873, 23: 121040, 24: 135340, 25: 148497, 26: 165882, 27: 177003, 28: 185152, 29: 187492, 30: 204544, 31: 183114, 32: 183603, 33: 177381, 34: 174403, 35: 147153, 36: 129541, 37: 113794, 38: 99679, 39: 79646, 40: 59481, 41: 46282, 42: 36100, 43: 26546, 44: 17533, 45: 15672, 46: 13709, 47: 7774, 48: 1256, 49: 5445, 50: 955, 51: 118, 52: 1, 53: 125}, 
              'atom_types': {0: 51915415, 1: 5, 2: 2, 3: 17730, 4: 35554802, 5: 5667122, 6: 4981302, 7: 561570, 8: 2, 9: 33336, 10: 40407, 11: 506659, 12: 310138, 13: 3, 14: 2, 15: 4, 16: 4, 17: 4, 18: 369, 19: 299, 20: 1459, 21: 36399}, 
              'colors_dic': ['#FFFFFF99', 'C2', 'C7', 'C0', 'C3', 'C1', 'C5', 'C6', 'C4', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20'], 'radius_dic': [0.3, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6], 'with_h': True}


atomic_number = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
                 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
                 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
                 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
                 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
                 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
                 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
                 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
                 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
                 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
                 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
                 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118}

pcq_data = build_geom_dataset.PCQM4MV2_Dihedral2(
    root='/data/protein/SKData/Denoise_Data/pcq', sdf_path='mol_iter_all.pickle', dihedral_angle_noise_scale=2, position_noise_scale=0.04, composition=True, decay=False, decay_coe=0.2, pre_transform=None, addh=True, new_transform=None,
)


pcq_len = len(pcq_data)

n_nodes = {}

atomic_dict = {}

def save_dict_to_yaml(data, filename):
    """
    Save a dictionary to a YAML file.
    """
    with open(filename, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

def atomic_number_to_symbol(atomic_number):
    """
    Convert atomic number to atomic symbol using RDKit.
    """
    symbol = Chem.GetPeriodicTable().GetElementSymbol(atomic_number)
    return symbol

# for i in tqdm(range(pcq_len)):
#     data_item = pcq_data[torch.tensor(i)]
#     atom_num = data_item.pos.shape[0]
#     if atom_num in n_nodes:
#         n_nodes[atom_num] += 1
#     else:
#         n_nodes[atom_num] = 1
    
#     for a_number in data_item.z:
#         atomic_symbol = atomic_number_to_symbol(a_number.item())
#         if atomic_symbol in atomic_dict:
#             atomic_dict[atomic_symbol] += 1
#         else:
#             atomic_dict[atomic_symbol] = 1
        
    
#     # print(pcq_data[torch.tensor(i)])
#     # print('xxx')
# save_dict_to_yaml(atomic_dict, 'atomic_dict.yaml')

# save_dict_to_yaml(n_nodes, 'n_nodes.yaml')

def parse_yaml(yaml_file):
    with open(yaml_file, "r") as yaml_file:
        data = yaml.safe_load(yaml_file)
    return data

n_nodes = parse_yaml('n_nodes.yaml')
atomic_dict = parse_yaml('atomic_dict.yaml')

# 'atom_encoder': {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 'Si': 7,
#     'P': 8, 'S': 9, 'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15},
# 'atomic_nb': [1,  5,  6,  7,  8,  9, 13, 14, 15, 16, 17, 33, 35, 53, 80, 83],
# 'atom_decoder': ['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I', 'Hg', 'Bi'],
# 'atom_types':{0: 143905848, 1: 290, 2: 129988623, 3: 20266722, 4: 21669359, 5: 1481844, 6: 1,
#                   7: 250, 8: 36290, 9: 3999872, 10: 1224394, 11: 4, 12: 298702, 13: 5377, 14: 13, 15: 34},
atom_encoder = {}
atomic_nb = []
atom_decoder = []
atom_types = {}

cnt = 0
for atom_sym in atomic_number:
    if atom_sym in atomic_dict:
        atom_decoder.append(atom_sym)
        atomic_nb.append(atomic_number[atom_sym])
        
        atom_encoder[atom_sym] = cnt
        atom_types[cnt] = atomic_dict[atom_sym]
        cnt += 1
        
        
pcq_with_h['atom_decoder'] = atom_decoder
pcq_with_h['atomic_nb'] = atomic_nb
pcq_with_h['atom_encoder'] = atom_encoder
pcq_with_h['max_n_nodes'] = max(n_nodes, key=n_nodes.get)
pcq_with_h['n_nodes'] = n_nodes
pcq_with_h['atom_types'] = atom_types

print(pcq_with_h)


allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'B': 3, 'Al': 3,
                 'Si': 4, 'P': [3, 5],
                 'S': 4, 'Cl': 1, 'As': 3, 'Br': 1, 'I': 1, 'Hg': [1, 2],
                 'Bi': [3, 5]}
# for ele in pcq_with_h['atom_decoder']:
#     if ele not in allowed_bonds:
#         print(ele)

allowed_bonds2 = {'He': 0, 'Be': 2,
'Mg': 2, 'Ar': 0, 'Ca': 2, 'Ti': [2, 3, 4], 'Zn': 2, 'Ga':[2, 3], 'Ge': [2, 4], 'Se': [2, 4 ,6]}



# for atom_symbol in pcq_with_h['atom_decoder']:
#     carbon_atom = Chem.Atom(atom_symbol)
#     # carbon_atom.calcExplicitValence()
#     total_valence = carbon_atom.GetTotalValence()

# import lmdb
# import pickle
# from tqdm import tqdm

# root = '/data/protein/SKData/Denoise_Data/pcq'
# MOL_LST = lmdb.open(f'{root}/MOL_LMDB', readonly=True, subdir=True, lock=False)

# txn = MOL_LST.begin()
# _keys = list(txn.cursor().iternext(values=False))


# valence_dict = {}
# pcq_len = len(_keys)

# for idx in tqdm(range(pcq_len)):
#     ky = str(idx).encode()
#     serialized_data = MOL_LST.begin().get(ky)
#     mol = pickle.loads(serialized_data)
#     for i in range(mol.GetNumAtoms()):
#         atom = mol.GetAtomWithIdx(i)
#         element = atom.GetSymbol()
#         # total_degree = atom.GetTotalDegree()
#         valence = atom.GetTotalValence()
#         if element in valence_dict:
#             valence_dict[element].add(valence)
#         else:
#             valence_dict[element] = set([valence])
#     # print('load mol')

# save_dict_to_yaml(valence_dict, 'valence_dict.yaml')
    
pass