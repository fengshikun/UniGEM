from build_geom_dataset import PCQM4MV2_Dihedral2
from tqdm import tqdm
import torch

pcq_data = PCQM4MV2_Dihedral2(root='/data/protein/SKData/Denoise_Data/pcq', sdf_path='mol_iter_all.pickle'
, dihedral_angle_noise_scale=2, position_noise_scale=0.04, composition=True, addh=True)



len_data = len(pcq_data)


property_lst = []

for i in tqdm(range(len_data)):
    data = pcq_data[torch.tensor([i])]
    property_lst.append(data.properties.cpu().numpy())
    
# statistic the properties mean and std, and save it to pkl file
import pickle
import numpy as np
property_lst = np.array(property_lst)
property_mean = np.mean(property_lst, axis=0)
property_std = np.std(property_lst, axis=0)
property_statistic = {'mean': property_mean, 'std': property_std}
with open('pcqm4m_properties_statistic.pkl', 'wb') as f:
    pickle.dump(property_statistic, f)
