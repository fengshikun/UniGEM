from rdkit import Chem
from rdkit.Chem import rdMolTransforms
import copy
from collections import defaultdict
import collections
import random
import numpy as np
import math
import torch
from rdkit.Geometry import Point3D

def get_torsions(mol_list):
    atom_counter = 0
    torsionList = []
    dihedralList = []
    for m in mol_list:
        torsionSmarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
        torsionQuery = Chem.MolFromSmarts(torsionSmarts)
        matches = m.GetSubstructMatches(torsionQuery)
        conf = m.GetConformer()
        for match in matches:
            idx2 = match[0]
            idx3 = match[1]
            bond = m.GetBondBetweenAtoms(idx2, idx3)
            jAtom = m.GetAtomWithIdx(idx2)
            kAtom = m.GetAtomWithIdx(idx3)
            for b1 in jAtom.GetBonds():
                if (b1.GetIdx() == bond.GetIdx()):
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)
                for b2 in kAtom.GetBonds():
                    if ((b2.GetIdx() == bond.GetIdx())
                            or (b2.GetIdx() == b1.GetIdx())):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    # skip 3-membered rings
                    if (idx4 == idx1):
                        continue
                    # skip torsions that include hydrogens
                    #                     if ((m.GetAtomWithIdx(idx1).GetAtomicNum() == 1)
                    #                         or (m.GetAtomWithIdx(idx4).GetAtomicNum() == 1)):
                    #                         continue
                    if m.GetAtomWithIdx(idx4).IsInRing():
                        torsionList.append(
                            (idx4 + atom_counter, idx3 + atom_counter, idx2 + atom_counter, idx1 + atom_counter))
                        break
                    else:
                        torsionList.append(
                            (idx1 + atom_counter, idx2 + atom_counter, idx3 + atom_counter, idx4 + atom_counter))
                        break
                break

        atom_counter += m.GetNumAtoms()
    return torsionList

def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)


def GetDihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])

def apply_changes(mol, values, rotable_bonds):
    opt_mol = copy.deepcopy(mol)
    #     opt_mol = add_rdkit_conformer(opt_mol)

    # apply rotations
    [SetDihedral(opt_mol.GetConformer(), rotable_bonds[r], values[r]) for r in range(len(rotable_bonds))]

    #     # apply transformation matrix
    #     rdMolTransforms.TransformConformer(opt_mol.GetConformer(), GetTransformationMatrix(values[:6]))

    return opt_mol



def GetBondLength(conf, atom_idx):
    return rdMolTransforms.GetBondLength(conf, atom_idx[0], atom_idx[1])

def SetBondLength(conf, atom_idx, new_vale):
    return rdMolTransforms.SetBondLength(conf, atom_idx[0], atom_idx[1], new_vale)

def GetAngle(conf, atom_idx):
    return rdMolTransforms.GetAngleDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2])

def SetAngle(conf, atom_idx, new_vale):
    return rdMolTransforms.SetAngleDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], new_vale)


def apply_changes_bond_length(mol, values, bond_idx):
    opt_mol = copy.deepcopy(mol)
    [SetBondLength(opt_mol.GetConformer(), bond_idx[r], values[r]) for r in range(len(bond_idx))]
    return opt_mol

def apply_changes_angle(mol, values, bond_idx):
    opt_mol = copy.deepcopy(mol)
    [SetAngle(opt_mol.GetConformer(), bond_idx[r], values[r]) for r in range(len(bond_idx))]

    return opt_mol

# input: mol (rdkit) object; rotate_bonds list
# return: rotate_bonds order(list, each rotate bond maybe reverse), and depth
def get_rotate_order_info(mol, rotate_bonds):
    cut_bonds_set = []
    for rb in rotate_bonds:
        cut_bonds_set.append([rb[1], rb[2]])
    left_bond_set = []
    
    bond_set = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bond_set.append([a1, a2])
    for ele in bond_set:
        if [ele[0], ele[1]] not in cut_bonds_set and \
            [ele[1], ele[0]] not in cut_bonds_set:
                left_bond_set.append(ele)
    graph = defaultdict(list)

    for x, y in left_bond_set:
        graph[x].append(y)
        graph[y].append(x)
    visited = set()
    mol_num = mol.GetNumAtoms()
    labels = [-1 for _ in range(mol_num)]

    rg_nodes = defaultdict(set) # key is the node idx of rigid-node graph, value is the indices of corresponding node in original mol graph.

    def dfs(i, lb=-1):
        visited.add(i)
        labels[i] = lb
        rg_nodes[lb].add(i)
        for j in graph[i]:
            if j not in visited:
                dfs(j, lb)

    lb = 0
    for i in range(mol_num):
        if i not in visited:
            dfs(i, lb)
            lb += 1
    # construct rigid-nodes graph 
    rg_graph = defaultdict(list)

    edge_rotate_bond_dict = {} # key is edge of rigid-nodes graph, value is the idx of rotate bonds

    for idx, rb in enumerate(cut_bonds_set):
        rg_edge = []
        # print('rb is {}'.format(rb))
        for key in rg_nodes:
            if rb[0] in rg_nodes[key]:
                rg_edge.append(key)
            if rb[1] in rg_nodes[key]:
                rg_edge.append(key)
            if len(rg_edge) == 2:
                edge_rotate_bond_dict[str(rg_edge)] = idx
                edge_rotate_bond_dict[str([rg_edge[1], rg_edge[0]])] = idx
                break
        # print(rg_edge)
        rg_graph[rg_edge[0]].append(rg_edge[1])
        rg_graph[rg_edge[1]].append(rg_edge[0])


    # add the node number into rg_graph:
    rg_graph_w_s = {}
    for rg in rg_graph:
        rg_graph_w_s[rg] = [rg_graph[rg], len(rg_nodes[rg])]
    
    def bfs(graph, root, edge_rotate_bond_dict, rotable_bonds, rg_nodes):
        seen, queue = {root}, collections.deque([(root, 0, -1)]) # vertex, level, rotate_bond_idx
        visit_order = []
        levels = []
        rotate_idx_lst = []
        rotable_bonds_order_lst = []
        while queue:
            vertex, level, rotate_bond_idx = queue.popleft()
            visit_order.append(vertex)
            levels.append(level)
            rotate_idx_lst.append(rotate_bond_idx)
            for node in graph.get(vertex, []):
                if node not in seen:
                    seen.add(node)

                    # [vetex, node] as edge
                    rotate_bond_idx = edge_rotate_bond_dict[str([vertex, node])]
                    rotate_bond = list(rotable_bonds[rotate_bond_idx])

                
                    #vetext--> node , check the rotate_bond reverse or not
                    if rotate_bond[0] not in rg_nodes[vertex] and rotate_bond[1] not in rg_nodes[vertex]: # reverse
                        rotate_bond.reverse()
                    
                    rotable_bonds_order_lst.append(rotate_bond)
                    queue.append((node, level + 1, rotate_bond_idx))
        return rotable_bonds_order_lst, levels[1:]
    
    root_idx = sorted(rg_graph_w_s.items(), key=lambda kv: (len(kv[1][0]), kv[1][1]))[0][0]
    return bfs(rg_graph, root_idx, edge_rotate_bond_dict, rotate_bonds, rg_nodes)


# method to get all bond, angle and dihedral angle vec points

def get_2d_gem(mol):
    # bond  (i, j)
    edge_idx = []
    edge_feat = []
    for bond in mol.GetBonds():
        i_idx = bond.GetBeginAtomIdx()
        j_idx = bond.GetEndAtomIdx()
        edge_idx.append([i_idx, j_idx])
    # NOTE: use from ogb.utils.features import bond_to_feature_vector to get bond featrue

    # angle (i, j, k)
    angle_idx = []
    for atom in mol.GetAtoms(): # j
        j_idx = atom.GetIdx()
        # atom_symbol = atom.GetSymbol()
        atom_degree = atom.GetDegree()
        if atom_degree >= 2:
            # get neighbors
            neighbors = atom.GetNeighbors()
            neb_lst = []
            for neb in neighbors:
                neb_lst.append(neb.GetIdx())

            neb_len = len(neb_lst)
            for i in range(neb_len):
                for k in range(i+1, neb_len):
                    angle_idx.append([neb_lst[i], j_idx, neb_lst[k]])
    
    # dihedral idx: (i, j, k, l), only return (j, k)
    dihedral_idx = []
    for bond in mol.GetBonds():
        j_idx = bond.GetBeginAtomIdx()
        k_idx = bond.GetEndAtomIdx()

        j_atom = mol.GetAtomWithIdx(j_idx)
        j_atom_degree = j_atom.GetDegree()
        k_atom = mol.GetAtomWithIdx(k_idx)
        k_atom_degree = k_atom.GetDegree()

        if j_atom_degree < 2 or k_atom_degree < 2: # cannot compose a dihedral angle
            continue

        # get neibors
        j_neighbors = j_atom.GetNeighbors()
        j_neb_lst = []
        for neb in j_neighbors:
            j_neb_lst.append(neb.GetIdx())
        j_neb_lst.remove(k_idx)

        k_neighbors = k_atom.GetNeighbors()
        k_neb_lst = []
        for neb in k_neighbors:
            k_neb_lst.append(neb.GetIdx())
        k_neb_lst.remove(j_idx)

        # random pick one neighbor from j and k, taken as i, l
        # i_idx = random.choice(j_neb_lst)
        # l_idx = random.choice(k_neb_lst)
        # dihedral_idx.append([i_idx, j_idx, k_idx, l_idx])
        for i_idx in j_neb_lst:
            for l_idx in k_neb_lst:
                dihedral_idx.append([i_idx, j_idx, k_idx, l_idx])
    
    return edge_idx, angle_idx, dihedral_idx


def get_info_by_gem_idx(org_mol, edge_idx, angle_idx, dihedral_idx):
    bond_len_lst = []
    conf = org_mol.GetConformer()
    for b_idx in edge_idx:
        bond_len_lst.append(GetBondLength(conf, b_idx))
    
    # angle
    angle_lst = []
    for a_idx in angle_idx:
        angle_lst.append(GetAngle(conf, a_idx))

    # dihedra 
    dihedral_lst = []
    for d_idx in dihedral_idx:
        dihedral_lst.append(GetDihedral(conf, d_idx))
    
    return np.array(bond_len_lst), np.array(angle_lst), np.array(dihedral_lst)




def get_info_by_gem_idx2(org_mol, edge_idx, angle_idx, dihedral_idx):
    bond_len_lst = []
    conf = org_mol.GetConformer()
    for b_idx in edge_idx:
        bond_len_lst.append(GetBondLength(conf, b_idx))
    
    # angle
    angle_lst = []
    for a_idx in angle_idx:
        angle_lst.append(getAngle_new(conf, a_idx))

    # dihedra 
    dihedral_lst = []
    for d_idx in dihedral_idx:
        dihedral_lst.append(GetDihedral(conf, d_idx))
    
    return np.array(bond_len_lst), np.array(angle_lst), np.array(dihedral_lst)


def filter_nan(idx_array, noise_array):
    valid_idx = ~np.isnan(noise_array)
    return idx_array[valid_idx], noise_array[valid_idx]

def check_in_samering(idx_i, idx_j, ring_array):
    for ring in ring_array:
        if idx_i in ring and idx_j in ring:
            return True
    return False

def concat_idx_label(edge_idx, angle_idx, dihedral_idx, noise_bond_len_label, noise_angle_label, noise_dihedral_label):
    edge_idx = np.array(edge_idx)
    angle_idx = np.array(angle_idx)
    dihedral_idx = np.array(dihedral_idx)

    edge_idx , noise_bond_len_label = filter_nan(edge_idx, noise_bond_len_label)
    angle_idx , noise_angle_label = filter_nan(angle_idx, noise_angle_label)
    dihedral_idx, noise_dihedral_label = filter_nan(dihedral_idx, noise_dihedral_label)

    
    # handle dihedral noise
    # noise_dihedral_label[noise_dihedral_label > 180] -= 360
    # noise_dihedral_label[noise_dihedral_label < -180] += 360

    # try:
    if edge_idx.size == 0:
        edge_res = np.array([])
    else:
        edge_res = np.concatenate([edge_idx, noise_bond_len_label.reshape([-1, 1])], axis=1)
    if angle_idx.size == 0:
        angle_res = np.array([])
    else:
        angle_res = np.concatenate([angle_idx, noise_angle_label.reshape([-1, 1])], axis=1)
    if dihedral_idx.size == 0:
        dihedral_res = np.array([])
    else:
        dihedral_res = np.concatenate([dihedral_idx, noise_dihedral_label.reshape([-1, 1])], axis=1)
    # except Exception as e:
    #     print(e)
    return edge_res, angle_res, dihedral_res


def wiki_dihedral_torch(pos, atomidx):
# def wiki_dihedral_torch(p0, p1, p2, p3):
    """formula from Wikipedia article on "Dihedral angle"; formula was removed
    from the most recent version of article (no idea why, the article is a
    mess at the moment) but the formula can be found in at this permalink to
    an old version of the article:
    https://en.wikipedia.org/w/index.php?title=Dihedral_angle&oldid=689165217#Angle_between_three_vectors
    uses 1 sqrt, 3 cross products"""
    # p0 = p[0]
    # p1 = p[1]
    # p2 = p[2]
    # p3 = p[3]

    p0, p1, p2, p3 = pos[atomidx[:, 0]], pos[atomidx[:, 1]], pos[atomidx[:, 2]], pos[atomidx[:,3]]

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b0xb1 = torch.cross(b0, b1)
    b1xb2 = torch.cross(b2, b1)

    b0xb1_x_b1xb2 = torch.cross(b0xb1, b1xb2)

    # y = np.dot(b0xb1_x_b1xb2, b1)*(1.0/np.linalg.norm(b1))
    # x = np.dot(b0xb1, b1xb2)

    y = (b0xb1_x_b1xb2 * b1).sum(dim=-1) / torch.norm(b1, dim=-1)
    x = (b0xb1 * b1xb2).sum(dim=-1)

    return torch.rad2deg(torch.atan2(y, x))


def getAngle_torch(pos, idx):
    # Calculate angles. 0 to pi
    # idx: i, j, k
    pos_ji = pos[idx[:, 0]] - pos[idx[:, 1]]
    pos_jk = pos[idx[:, 2]] - pos[idx[:, 1]]
    a = (pos_ji * pos_jk).sum(dim=-1) # cos_angle * |pos_ji| * |pos_jk|
    b = torch.cross(pos_ji, pos_jk).norm(dim=-1) # sin_angle * |pos_ji| * |pos_jk|
    angle = torch.atan2(b, a)
    return torch.rad2deg(angle)

def getAngle_new(conf, atomidx):
    i, j, k  = np.array(conf.GetAtomPosition(atomidx[0])), np.array(conf.GetAtomPosition(atomidx[1])), np.array(conf.GetAtomPosition(atomidx[2]))
    pos_ji = i - j 
    pos_jk = k - j
    a = (pos_ji * pos_jk).sum(axis=-1) # cos_angle * |pos_ji| * |pos_jk|
    cross_vec = np.cross(pos_ji, pos_jk)
    b = np.linalg.norm(cross_vec)
 # sin_angle * |pos_ji| * |pos_jk|
    angle = np.arctan2(b, a)
    # NOTE: no sign
    # zero_mask = (cross_vec == 0) 
    # if zero_mask.sum() == 0:
    #     if (cross_vec < 0).sum() >= 2:
    #         angle *= -1
    # else: # has zero
    #     angle *= np.sign(cross_vec[~zero_mask][0])


    return angle * 57.3


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def getTorsionNew(conf, atomidx):
    # Calculate torsions.
    # i, j, k, l
    idx_i, idx_j, idx_k, idx_l = np.array(conf.GetAtomPosition(atomidx[0])), np.array(conf.GetAtomPosition(atomidx[1])), np.array(conf.GetAtomPosition(atomidx[2])), np.array(conf.GetAtomPosition(atomidx[3]))
    pos_ji = idx_i - idx_j
    pos_jk = idx_k - idx_j
    pos_kj = -pos_jk
    pos_kl = idx_k - idx_l

    plane1 = np.cross(pos_ji, pos_jk)
    plane2 = np.cross(pos_kl, pos_kj)

    # return np.rad2deg(angle_between(plane1, plane2))

    a = (plane1 * plane2).sum(axis=-1) # cos_angle * |pos_ji| * |pos_jk|
    cross_vec = np.cross(plane1, plane2)
    b = np.linalg.norm(cross_vec)
 # sin_angle * |pos_ji| * |pos_jk|
    angle = np.arctan2(b, a)
    # angle = np.pi - angle
    # if angle <= 0:
    #     angle += (2 * np.pi)    

    return np.rad2deg(angle)

    # pos_j0 = pos[idx_k_t] - pos[idx_j_t]
    # pos_ji = pos[idx_i_t] - pos[idx_j_t]
    # pos_jk = pos[idx_k_n] - pos[idx_j_t]

    dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
    plane1 = torch.cross(pos_ji, pos_j0)
    plane2 = torch.cross(pos_ji, pos_jk)
    a = (plane1 * plane2).sum(dim=-1) # cos_angle * |plane1| * |plane2|
    b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji 
    torsion1 = torch.atan2(b, a) # -pi to pi
    torsion1[torsion1<=0]+=2*PI # 0 to 2pi

def wiki_dihedral(conf, atomidx):
    """formula from Wikipedia article on "Dihedral angle"; formula was removed
    from the most recent version of article (no idea why, the article is a
    mess at the moment) but the formula can be found in at this permalink to
    an old version of the article:
    https://en.wikipedia.org/w/index.php?title=Dihedral_angle&oldid=689165217#Angle_between_three_vectors
    uses 1 sqrt, 3 cross products"""
    # p0 = p[0]
    # p1 = p[1]
    # p2 = p[2]
    # p3 = p[3]

    p0, p1, p2, p3 = np.array(conf.GetAtomPosition(atomidx[0])), np.array(conf.GetAtomPosition(atomidx[1])), np.array(conf.GetAtomPosition(atomidx[2])), np.array(conf.GetAtomPosition(atomidx[3]))

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b0xb1 = np.cross(b0, b1)
    b1xb2 = np.cross(b2, b1)

    b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)

    y = np.dot(b0xb1_x_b1xb2, b1)*(1.0/np.linalg.norm(b1))
    x = np.dot(b0xb1, b1xb2)

    return np.degrees(np.arctan2(y, x))


def add_equi_keep_noise2(opt_mol, bond_var=0.04, angle_var=0.09, torsion_var=0.69, coord_var=0.04, add_ring_noise=False):
    mol = copy.deepcopy(opt_mol)
    conf = mol.GetConformer()
    edge_idx, angle_idx, dihedral_idx = get_2d_gem(mol)
    if add_ring_noise:
        atom_num = mol.GetNumAtoms()
        pos_coords = np.zeros((atom_num, 3), dtype=np.float32)
        # pos_noise_coords = new_mol.GetConformer().GetPositions()
        for idx in range(atom_num):
            c_pos = conf.GetAtomPosition(idx)
            pos_coords[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]
        ring_mask = []
        for atom in mol.GetAtoms(): # j
            if atom.IsInRing():
                ring_mask.append(True)
            else:
                ring_mask.append(False)
        noise = torch.randn_like(torch.tensor(pos_coords)) * coord_var
        data_noise = pos_coords + noise.numpy()
        # set back to conf
        for i in range(atom_num):
            if ring_mask[i]: # only add ring noise
                x,y,z = data_noise[i]
                conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    
        # keep record of bond and angle of original mol
        bond_len_lst, angle_lst, dihedral_lst = get_info_by_gem_idx2(opt_mol, edge_idx, angle_idx, dihedral_idx)
        # noise mol
        bond_len_noise_lst, angle_noise_lst, dihedral_noise_lst = get_info_by_gem_idx2(mol, edge_idx, angle_idx, dihedral_idx)
        
        bond_len_noise_lst -= bond_len_lst
        angle_noise_lst -= angle_lst
        dihedral_noise_lst -= dihedral_lst
    else:
        bond_len_noise_lst = np.zeros(len(edge_idx), dtype=np.float32)
        angle_noise_lst = np.zeros(len(angle_idx), dtype=np.float32)
        dihedral_noise_lst = np.zeros(len(dihedral_idx), dtype=np.float32)
    
    bond_label_lst, angle_label_lst, dihedral_label_lst = concat_idx_label(edge_idx, angle_idx, dihedral_idx, bond_len_noise_lst, angle_noise_lst, dihedral_noise_lst)

    # build dict for the bond
    edge_dict = {}
    angle_dict = {}
    dihedral_dict = {}
    edge_idx = np.array(edge_idx)
    angle_idx = np.array(angle_idx)
    dihedral_idx = np.array(dihedral_idx)
    for i, ele_idx in enumerate(edge_idx[:, :2]):
        key = '_'.join([str(ele) for ele in ele_idx])
        edge_dict[key] = i
    
    if angle_idx.size != 0:
        for i, ele_idx in enumerate(angle_idx[:, :3]):
            key = '_'.join([str(ele) for ele in ele_idx])
            angle_dict[key] = i

    if dihedral_idx.size != 0:
        for i, ele_idx in enumerate(dihedral_idx[:, 2:4]):
            key = '_'.join([str(ele) for ele in ele_idx])
            dihedral_dict[key] = i


    # get ring info
    ring_array = []
    for ring in mol.GetRingInfo().AtomRings():
        ring_array.append(ring)


    # add noise manually
    for bond in mol.GetBonds():
        i_idx = bond.GetBeginAtomIdx()
        j_idx = bond.GetEndAtomIdx()
        # if mol.GetAtomWithIdx(i_idx).IsInRing() and mol.GetAtomWithIdx(j_idx).IsInRing():
        if check_in_samering(i_idx, j_idx, ring_array):
            continue

        org_bond_len = GetBondLength(conf, [i_idx, j_idx])
        # add gaussian noise:
        noise_bond_len = np.random.normal(loc=org_bond_len, scale=bond_var)
        # set bond_length
        SetBondLength(conf, [i_idx, j_idx], noise_bond_len)
        # noise 
        key = f'{i_idx}_{j_idx}'
        if key in edge_dict:
            noise = noise_bond_len - org_bond_len
            bond_label_lst[edge_dict[key]][-1] += noise


    # angle noise
    for atom in mol.GetAtoms(): # j
        j_idx = atom.GetIdx()
        # atom_symbol = atom.GetSymbol()
        atom_degree = atom.GetDegree()
        if atom_degree >= 2:
            # get neighbors
            neighbors = atom.GetNeighbors()
            neb_lst = []
            for neb in neighbors:
                neb_lst.append(neb.GetIdx())

            if mol.GetAtomWithIdx(j_idx).IsInRing(): # if j in ring, must pick one neb in ring as i
                for n_idx in neb_lst:
                    # if mol.GetAtomWithIdx(n_idx).IsInRing():
                    if check_in_samering(j_idx, n_idx, ring_array):
                        i_idx = n_idx
                        break
            else:
                # j not in ring, random pick one as i
                i_idx = random.choice(neb_lst)
            
            neb_lst.remove(i_idx)
            # iterate k
            for k_idx in neb_lst:
                # judge (i, j) and (j, k) in ring:
                # if mol.GetAtomWithIdx(i_idx).IsInRing() and mol.GetAtomWithIdx(j_idx).IsInRing() and mol.GetAtomWithIdx(k_idx).IsInRing():
                #     continue
                if check_in_samering(i_idx, j_idx, ring_array) and check_in_samering(j_idx, k_idx, ring_array):
                    continue
                # add angle noise to (i, j, k)
                org_angle = getAngle_new(conf, [i_idx, j_idx, k_idx]) 
                if math.isnan(org_angle): # may be nan
                    continue
                # add noise
                # FIX by change the unit to radian from degree
                noise = np.random.normal(scale=angle_var)
                noise_angle = org_angle + noise * 57.3 

                # noise_angle = np.random.normal(loc=org_angle, scale=angle_var)
                if org_angle < 0:
                    set_angle = -noise_angle
                else:
                    set_angle = noise_angle
                SetAngle(conf, [i_idx, j_idx, k_idx], set_angle)


                for a_idx in angle_idx:
                    a_idx = [int(ele) for ele in a_idx]
                    noise_angle_tmp = getAngle_new(conf, a_idx)
                    org_angle_tmp = getAngle_new(opt_mol.GetConformer(), a_idx)
                    if np.abs(org_angle_tmp - noise_angle_tmp) > 200:
                        print('haha')

                noise_angle = getAngle_new(conf, [i_idx, j_idx, k_idx]) 
                # if noise_angle - org_angle > 200:
                #     print('hahha')
                
                # if j_idx == 0 and (19 in [i_idx, k_idx]) and (20 in [i_idx, k_idx]):
                #     print('hahah')
    

                # angle_label_lst.append([i_idx, j_idx, k_idx, noise_angle - org_angle])
    # Get Angle:
    # _, angle_noise_lst, _ = get_info_by_gem_idx2(mol, edge_idx, angle_idx, dihedral_idx)
    angle_noise_lst = []
    for a_idx in angle_idx:
        a_idx = [int(ele) for ele in a_idx]
        angle_noise_lst.append(getAngle_new(conf, a_idx))

    angle_label_lst[:, -1] = np.array(angle_noise_lst) - angle_lst

    # dihedral angle(rotatable or not) [i, j, k, l]
    # get the all the rotatable angel idx
    rotable_bonds = get_torsions([mol]) # format like [(0, 5, 10, 7), (1, 6, 12, 11), (6, 12, 11, 4)]
    
    rotable_sets = set([])
    for rb in rotable_bonds:
        rotable_sets.add(f'{rb[1]}_{rb[2]}')
        rotable_sets.add(f'{rb[2]}_{rb[1]}')

    r_dihedral_label_lst = [] # [i, j, k, l, delta_angle]
    r_rotate_dihedral_label_lst = []
    # for bond in mol.GetBonds():

        # is_rotate = False

        # j_idx = bond.GetBeginAtomIdx()
        # k_idx = bond.GetEndAtomIdx()
        # # check (j_idx, k_idx) in ring or not
        # if mol.GetAtomWithIdx(j_idx).IsInRing() and mol.GetAtomWithIdx(k_idx).IsInRing():
        #     continue
        
        # j_atom = mol.GetAtomWithIdx(j_idx)
        # j_atom_degree = j_atom.GetDegree()
        # k_atom = mol.GetAtomWithIdx(k_idx)
        # k_atom_degree = k_atom.GetDegree()

        # if j_atom_degree < 2 or k_atom_degree < 2: # cannot compose a dihedral angle
        #     continue

        # # get neibors
        # j_neighbors = j_atom.GetNeighbors()
        # j_neb_lst = []
        # for neb in j_neighbors:
        #     j_neb_lst.append(neb.GetIdx())
        # j_neb_lst.remove(k_idx)

        # k_neighbors = k_atom.GetNeighbors()
        # k_neb_lst = []
        # for neb in k_neighbors:
        #     k_neb_lst.append(neb.GetIdx())
        # k_neb_lst.remove(j_idx)

        # # random pick one neighbor from j and k, taken as i, l
        # i_idx = random.choice(j_neb_lst)
        # l_idx = random.choice(k_neb_lst)

        # if f'{j_idx}_{k_idx}' in rotable_sets: # rotatable
        #     deh_var = torsion_var
        #     is_rotate = True
        # else:
        #     deh_var = angle_var
        #     is_rotate = False
        
        # # add noise
        # org_deh_angle = GetDihedral(conf, [i_idx, j_idx, k_idx, l_idx])
        # if math.isnan(org_deh_angle): # may be nan
        #     continue

        # # FIX by change the unit to radian from degree
        # noise = np.random.normal(scale=deh_var)
        # noise_deh_angle = org_deh_angle + noise * 57.3

        # # noise_deh_angle = np.random.normal(loc=org_deh_angle, scale=deh_var)
        # SetDihedral(conf, [i_idx, j_idx, k_idx, l_idx], noise_deh_angle)

        # # noise 
        # key = f'{j_idx}_{k_idx}'
        # if key in dihedral_dict:
        #     noise = noise_deh_angle - org_deh_angle
        #     dihedral_label_lst[dihedral_dict[key]][-1] += noise
        #     if is_rotate:
        #         r_dihedral_label_lst.append(dihedral_label_lst[dihedral_dict[key]])
        #     else:
        #         r_rotate_dihedral_label_lst.append(dihedral_label_lst[dihedral_dict[key]])

        # if is_rotate:
        #     rotate_dihedral_label_lst.append([i_idx, j_idx, k_idx, l_idx, noise_deh_angle - org_deh_angle])
        # else:
        #     dihedral_label_lst.append([i_idx, j_idx, k_idx, l_idx, noise_deh_angle - org_deh_angle])

    if angle_label_lst.size != 0:
        angle_label_lst = angle_label_lst[angle_label_lst[:,-1] != 0]

    return mol, bond_label_lst, angle_label_lst, r_dihedral_label_lst, r_rotate_dihedral_label_lst
    # return mol, bond_label_lst, angle_label_lst, dihedral_label_lst



def add_equi_keep_noise(opt_mol, bond_var=0.04, angle_var=0.09, torsion_var=0.69, coord_var=0.04, add_ring_noise=False):
    mol = copy.deepcopy(opt_mol)
    conf = mol.GetConformer()
    edge_idx, angle_idx, dihedral_idx = get_2d_gem(mol)
    if add_ring_noise:
        atom_num = mol.GetNumAtoms()
        pos_coords = np.zeros((atom_num, 3), dtype=np.float32)
        # pos_noise_coords = new_mol.GetConformer().GetPositions()
        for idx in range(atom_num):
            c_pos = conf.GetAtomPosition(idx)
            pos_coords[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]
        ring_mask = []
        for atom in mol.GetAtoms(): # j
            if atom.IsInRing():
                ring_mask.append(True)
            else:
                ring_mask.append(False)
        noise = torch.randn_like(torch.tensor(pos_coords)) * coord_var
        data_noise = pos_coords + noise.numpy()
        # set back to conf
        for i in range(atom_num):
            if ring_mask[i]: # only add ring noise
                x,y,z = data_noise[i]
                conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    
        # keep record of bond and angle of original mol
        bond_len_lst, angle_lst, dihedral_lst = get_info_by_gem_idx(opt_mol, edge_idx, angle_idx, dihedral_idx)
        # noise mol
        bond_len_noise_lst, angle_noise_lst, dihedral_noise_lst = get_info_by_gem_idx(mol, edge_idx, angle_idx, dihedral_idx)
        
        bond_len_noise_lst -= bond_len_lst
        angle_noise_lst -= angle_lst
        dihedral_noise_lst -= dihedral_lst
    else:
        bond_len_noise_lst = np.zeros(len(edge_idx), dtype=np.float32)
        angle_noise_lst = np.zeros(len(angle_idx), dtype=np.float32)
        dihedral_noise_lst = np.zeros(len(dihedral_idx), dtype=np.float32)
    
    bond_label_lst, angle_label_lst, dihedral_label_lst = concat_idx_label(edge_idx, angle_idx, dihedral_idx, bond_len_noise_lst, angle_noise_lst, dihedral_noise_lst)

    # build dict for the bond
    edge_dict = {}
    angle_dict = {}
    dihedral_dict = {}
    edge_idx = np.array(edge_idx)
    angle_idx = np.array(angle_idx)
    dihedral_idx = np.array(dihedral_idx)
    for i, ele_idx in enumerate(edge_idx[:, :2]):
        key = '_'.join([str(ele) for ele in ele_idx])
        edge_dict[key] = i
    
    if angle_idx.size != 0:
        for i, ele_idx in enumerate(angle_idx[:, :3]):
            key = '_'.join([str(ele) for ele in ele_idx])
            angle_dict[key] = i

    if dihedral_idx.size != 0:
        for i, ele_idx in enumerate(dihedral_idx[:, 2:4]):
            key = '_'.join([str(ele) for ele in ele_idx])
            dihedral_dict[key] = i


    # get ring info
    ring_array = []
    for ring in mol.GetRingInfo().AtomRings():
        ring_array.append(ring)


    # add noise manually
    for bond in mol.GetBonds():
        i_idx = bond.GetBeginAtomIdx()
        j_idx = bond.GetEndAtomIdx()
        # if mol.GetAtomWithIdx(i_idx).IsInRing() and mol.GetAtomWithIdx(j_idx).IsInRing():
        if check_in_samering(i_idx, j_idx, ring_array):
            continue

        org_bond_len = GetBondLength(conf, [i_idx, j_idx])
        # add gaussian noise:
        noise_bond_len = np.random.normal(loc=org_bond_len, scale=bond_var)
        # set bond_length
        SetBondLength(conf, [i_idx, j_idx], noise_bond_len)
        # noise 
        key = f'{i_idx}_{j_idx}'
        if key in edge_dict:
            noise = noise_bond_len - org_bond_len
            bond_label_lst[edge_dict[key]][-1] += noise


    # angle noise
    angle_del_idx = []
    for atom in mol.GetAtoms(): # j
        j_idx = atom.GetIdx()
        # atom_symbol = atom.GetSymbol()
        atom_degree = atom.GetDegree()
        if atom_degree >= 2:
            # get neighbors
            neighbors = atom.GetNeighbors()
            neb_lst = []
            for neb in neighbors:
                neb_lst.append(neb.GetIdx())

            if mol.GetAtomWithIdx(j_idx).IsInRing(): # if j in ring, must pick one neb in ring as i
                for n_idx in neb_lst:
                    # if mol.GetAtomWithIdx(n_idx).IsInRing():
                    if check_in_samering(j_idx, n_idx, ring_array):
                        i_idx = n_idx
                        break
            else:
                # j not in ring, random pick one as i
                i_idx = random.choice(neb_lst)
            
            neb_lst.remove(i_idx)
            # iterate k
            for k_idx in neb_lst:
                # judge (i, j) and (j, k) in ring:
                # if mol.GetAtomWithIdx(i_idx).IsInRing() and mol.GetAtomWithIdx(j_idx).IsInRing() and mol.GetAtomWithIdx(k_idx).IsInRing():
                #     continue
                if check_in_samering(i_idx, j_idx, ring_array) and check_in_samering(j_idx, k_idx, ring_array):
                    continue
                # add angle noise to (i, j, k)
                org_angle = GetAngle(conf, [i_idx, j_idx, k_idx])
                if math.isnan(org_angle): # may be nan
                    continue
                # add noise
                # FIX by change the unit to radian from degree
                noise = np.random.normal(scale=angle_var)
                noise_angle = org_angle + noise * 57.3 

                # noise_angle = np.random.normal(loc=org_angle, scale=angle_var)
                SetAngle(conf, [i_idx, j_idx, k_idx], noise_angle)
                
                # noise 
                key = f'{i_idx}_{j_idx}_{k_idx}'
                r_key = f'{k_idx}_{j_idx}_{i_idx}'
                if key in angle_dict:
                    delta_noise = noise_angle - org_angle
                    angle_label_lst[angle_dict[key]][-1] += delta_noise
                elif r_key in angle_dict:
                    delta_noise = noise_angle - org_angle
                    angle_label_lst[angle_dict[r_key]][-1] += delta_noise

                # erase others item contains j_idx, but no k_idx
                for idx, item in enumerate(angle_label_lst):
                    if item[1] == j_idx and (k_idx not in item[:-1]):
                        angle_del_idx.append(idx)
    
    angle_del_idx = list(set(angle_del_idx))
    angle_label_lst = np.delete(angle_label_lst, angle_del_idx, axis=0)

                # angle_label_lst.append([i_idx, j_idx, k_idx, noise_angle - org_angle])
    
    # dihedral angle(rotatable or not) [i, j, k, l]
    # get the all the rotatable angel idx
    rotable_bonds = get_torsions([mol]) # format like [(0, 5, 10, 7), (1, 6, 12, 11), (6, 12, 11, 4)]
    
    rotable_sets = set([])
    for rb in rotable_bonds:
        rotable_sets.add(f'{rb[1]}_{rb[2]}')
        rotable_sets.add(f'{rb[2]}_{rb[1]}')

    r_dihedral_label_lst = [] # [i, j, k, l, delta_angle]
    r_rotate_dihedral_label_lst = []
    # for bond in mol.GetBonds():

        # is_rotate = False

        # j_idx = bond.GetBeginAtomIdx()
        # k_idx = bond.GetEndAtomIdx()
        # # check (j_idx, k_idx) in ring or not
        # if mol.GetAtomWithIdx(j_idx).IsInRing() and mol.GetAtomWithIdx(k_idx).IsInRing():
        #     continue
        
        # j_atom = mol.GetAtomWithIdx(j_idx)
        # j_atom_degree = j_atom.GetDegree()
        # k_atom = mol.GetAtomWithIdx(k_idx)
        # k_atom_degree = k_atom.GetDegree()

        # if j_atom_degree < 2 or k_atom_degree < 2: # cannot compose a dihedral angle
        #     continue

        # # get neibors
        # j_neighbors = j_atom.GetNeighbors()
        # j_neb_lst = []
        # for neb in j_neighbors:
        #     j_neb_lst.append(neb.GetIdx())
        # j_neb_lst.remove(k_idx)

        # k_neighbors = k_atom.GetNeighbors()
        # k_neb_lst = []
        # for neb in k_neighbors:
        #     k_neb_lst.append(neb.GetIdx())
        # k_neb_lst.remove(j_idx)

        # # random pick one neighbor from j and k, taken as i, l
        # i_idx = random.choice(j_neb_lst)
        # l_idx = random.choice(k_neb_lst)

        # if f'{j_idx}_{k_idx}' in rotable_sets: # rotatable
        #     deh_var = torsion_var
        #     is_rotate = True
        # else:
        #     deh_var = angle_var
        #     is_rotate = False
        
        # # add noise
        # org_deh_angle = GetDihedral(conf, [i_idx, j_idx, k_idx, l_idx])
        # if math.isnan(org_deh_angle): # may be nan
        #     continue

        # # FIX by change the unit to radian from degree
        # noise = np.random.normal(scale=deh_var)
        # noise_deh_angle = org_deh_angle + noise * 57.3

        # # noise_deh_angle = np.random.normal(loc=org_deh_angle, scale=deh_var)
        # SetDihedral(conf, [i_idx, j_idx, k_idx, l_idx], noise_deh_angle)

        # # noise 
        # key = f'{j_idx}_{k_idx}'
        # if key in dihedral_dict:
        #     noise = noise_deh_angle - org_deh_angle
        #     dihedral_label_lst[dihedral_dict[key]][-1] += noise
        #     if is_rotate:
        #         r_dihedral_label_lst.append(dihedral_label_lst[dihedral_dict[key]])
        #     else:
        #         r_rotate_dihedral_label_lst.append(dihedral_label_lst[dihedral_dict[key]])

        # if is_rotate:
        #     rotate_dihedral_label_lst.append([i_idx, j_idx, k_idx, l_idx, noise_deh_angle - org_deh_angle])
        # else:
        #     dihedral_label_lst.append([i_idx, j_idx, k_idx, l_idx, noise_deh_angle - org_deh_angle])

    if angle_label_lst.size != 0:
        angle_label_lst = angle_label_lst[angle_label_lst[:,-1] != 0]

    return mol, bond_label_lst, angle_label_lst, r_dihedral_label_lst, r_rotate_dihedral_label_lst
    # return mol, bond_label_lst, angle_label_lst, dihedral_label_lst



def add_equi_noise(opt_mol, bond_var=0.04, angle_var=0.04, torsion_var=2, coord_var=0.04, add_ring_noise=False):
    # bond noise, find all bond add noise
    org_conf = opt_mol.GetConformer()
    mol = copy.deepcopy(opt_mol)
    conf = mol.GetConformer()
    if add_ring_noise:
        atom_num = mol.GetNumAtoms()
        pos_coords = np.zeros((atom_num, 3), dtype=np.float32)
        # pos_noise_coords = new_mol.GetConformer().GetPositions()
        for idx in range(atom_num):
            c_pos = conf.GetAtomPosition(idx)
            pos_coords[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]
        ring_mask = []
        for atom in mol.GetAtoms(): # j
            if atom.IsInRing():
                ring_mask.append(True)
            else:
                ring_mask.append(False)
        noise = torch.randn_like(torch.tensor(pos_coords)) * coord_var
        data_noise = pos_coords + noise.numpy()
        # set back to conf
        for i in range(atom_num):
            if ring_mask[i]: # only add ring noise
                x,y,z = data_noise[i]
                conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        



    bond_label_lst = [] # [i, j, delta_len]
    for bond in mol.GetBonds():
        i_idx = bond.GetBeginAtomIdx()
        j_idx = bond.GetEndAtomIdx()
        if mol.GetAtomWithIdx(i_idx).IsInRing() and mol.GetAtomWithIdx(j_idx).IsInRing():
            continue

        org_bond_len = GetBondLength(conf, [i_idx, j_idx])
        # add gaussian noise:
        noise_bond_len = np.random.normal(loc=org_bond_len, scale=bond_var)
        # set bond_length
        SetBondLength(conf, [i_idx, j_idx], noise_bond_len)
        bond_label_lst.append([i_idx, j_idx, noise_bond_len - org_bond_len])

    # angle noise
    angle_label_lst = [] # [i, j, k, delta_angle]
    for atom in mol.GetAtoms(): # j
        j_idx = atom.GetIdx()
        # atom_symbol = atom.GetSymbol()
        atom_degree = atom.GetDegree()
        if atom_degree >= 2:
            # get neighbors
            neighbors = atom.GetNeighbors()
            neb_lst = []
            for neb in neighbors:
                neb_lst.append(neb.GetIdx())

            if mol.GetAtomWithIdx(j_idx).IsInRing(): # if j in ring, must pick one neb in ring as i
                for n_idx in neb_lst:
                    if mol.GetAtomWithIdx(n_idx).IsInRing():
                        i_idx = n_idx
                        break
            else:
                # j not in ring, random pick one as i
                i_idx = random.choice(neb_lst)
            
            neb_lst.remove(i_idx)
            # iterate k
            for k_idx in neb_lst:
                # judge (i, j) and (j, k) in ring:
                if mol.GetAtomWithIdx(i_idx).IsInRing() and mol.GetAtomWithIdx(j_idx).IsInRing() and mol.GetAtomWithIdx(k_idx).IsInRing():
                    continue
                # add angle noise to (i, j, k)
                org_angle = GetAngle(org_conf, [i_idx, j_idx, k_idx])
                if math.isnan(org_angle): # may be nan
                    continue
                # add noise
                # FIX by change the unit to radian from degree
                noise = np.random.normal(scale=angle_var)
                noise_angle = org_angle + noise * 57.3

                # noise_angle = np.random.normal(loc=org_angle, scale=angle_var)
                SetAngle(conf, [i_idx, j_idx, k_idx], noise_angle)
                noise_angle_new = GetAngle(conf, [i_idx, j_idx, k_idx])
                # angle_label_lst.append([i_idx, j_idx, k_idx, noise_angle - org_angle])
                angle_label_lst.append([i_idx, j_idx, k_idx, noise_angle_new - org_angle])
    
    # dihedral angle(rotatable or not) [i, j, k, l]
    # get the all the rotatable angel idx
    rotable_bonds = get_torsions([mol]) # format like [(0, 5, 10, 7), (1, 6, 12, 11), (6, 12, 11, 4)]
    
    rotable_sets = set([])
    for rb in rotable_bonds:
        rotable_sets.add(f'{rb[1]}_{rb[2]}')
        rotable_sets.add(f'{rb[2]}_{rb[1]}')

    dihedral_label_lst = [] # [i, j, k, l, delta_angle]
    rotate_dihedral_label_lst = []
    # for bond in mol.GetBonds():

    #     is_rotate = False

    #     j_idx = bond.GetBeginAtomIdx()
    #     k_idx = bond.GetEndAtomIdx()
    #     # check (j_idx, k_idx) in ring or not
    #     if mol.GetAtomWithIdx(j_idx).IsInRing() and mol.GetAtomWithIdx(k_idx).IsInRing():
    #         continue
        
    #     j_atom = mol.GetAtomWithIdx(j_idx)
    #     j_atom_degree = j_atom.GetDegree()
    #     k_atom = mol.GetAtomWithIdx(k_idx)
    #     k_atom_degree = k_atom.GetDegree()

    #     if j_atom_degree < 2 or k_atom_degree < 2: # cannot compose a dihedral angle
    #         continue

    #     # get neibors
    #     j_neighbors = j_atom.GetNeighbors()
    #     j_neb_lst = []
    #     for neb in j_neighbors:
    #         j_neb_lst.append(neb.GetIdx())
    #     j_neb_lst.remove(k_idx)

    #     k_neighbors = k_atom.GetNeighbors()
    #     k_neb_lst = []
    #     for neb in k_neighbors:
    #         k_neb_lst.append(neb.GetIdx())
    #     k_neb_lst.remove(j_idx)

    #     # random pick one neighbor from j and k, taken as i, l
    #     i_idx = random.choice(j_neb_lst)
    #     l_idx = random.choice(k_neb_lst)

    #     if f'{j_idx}_{k_idx}' in rotable_sets: # rotatable
    #         deh_var = torsion_var
    #         is_rotate = True
    #     else:
    #         deh_var = angle_var
    #         is_rotate = False
        
    #     # add noise
    #     org_deh_angle = GetDihedral(conf, [i_idx, j_idx, k_idx, l_idx])
    #     if math.isnan(org_deh_angle): # may be nan
    #         continue

    #     # FIX by change the unit to radian from degree
    #     noise = np.random.normal(scale=deh_var)
    #     noise_deh_angle = org_deh_angle + noise * 57.3

    #     # noise_deh_angle = np.random.normal(loc=org_deh_angle, scale=deh_var)
    #     SetDihedral(conf, [i_idx, j_idx, k_idx, l_idx], noise_deh_angle)

    #     if is_rotate:
    #         rotate_dihedral_label_lst.append([i_idx, j_idx, k_idx, l_idx, noise_deh_angle - org_deh_angle])
    #     else:
    #         dihedral_label_lst.append([i_idx, j_idx, k_idx, l_idx, noise_deh_angle - org_deh_angle])
    
    bond_label_lst = np.array(bond_label_lst)
    angle_label_lst = np.array(angle_label_lst)
    return mol, bond_label_lst, angle_label_lst, dihedral_label_lst, rotate_dihedral_label_lst


# special NOTE:
# add_noise_type: 0, BAT noise
# add_noise_type: 1, Frad noise
# add_noise_type: 2, coordinate guassian noise


def transform_noise(data, position_noise_scale=0.04):
    noise = torch.randn_like(torch.tensor(data)) * position_noise_scale
    data_noise = data + noise.numpy()
    return data_noise

def add_equi_noise_new(opt_mol, bond_var=0.04, angle_var=0.143, torsion_var_r=2.8, torsion_var=0.41, coord_var=0.04, add_ring_noise=False, add_noise_type=0, mol_param=None, ky=0):
    # bond noise, find all bond add noise
    
    org_conf = opt_mol.GetConformer()
    mol = copy.deepcopy(opt_mol)
    conf = mol.GetConformer()
    
    atom_num = mol.GetNumAtoms()
    pos_coords = np.zeros((atom_num, 3), dtype=np.float32)
        # pos_noise_coords = new_mol.GetConformer().GetPositions()
    for idx in range(atom_num):
        c_pos = conf.GetAtomPosition(idx)
        pos_coords[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]
    
    
    if add_noise_type == 2: # gaussain noise
        data_noise = transform_noise(pos_coords)
        # set the data_noise back to the mol
        for i in range(atom_num):
            x,y,z = data_noise[i]
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        rotate_dihedral_label_lst = []
    elif add_noise_type == 1: # Frad noise
        rotable_bonds = get_torsions([mol])
        org_angle = []
        for rot_bond in rotable_bonds:
            org_angle.append(GetDihedral(mol.GetConformer(), rot_bond))
        org_angle = np.array(org_angle)        
        noise_angle = transform_noise(org_angle, position_noise_scale=2)
        new_mol = apply_changes(mol, noise_angle, rotable_bonds) # add the 
        coord_conf = new_mol.GetConformer()
        
        pos_noise_coords_angle = np.zeros((atom_num, 3), dtype=np.float32)
        # pos_noise_coords = new_mol.GetConformer().GetPositions()
        for idx in range(atom_num):
            c_pos = coord_conf.GetAtomPosition(idx)
            pos_noise_coords_angle[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]
        # add guassian noise
        pos_noise_coords = transform_noise(pos_noise_coords_angle, position_noise_scale=0.04)
        
        # set back to the mol
        opt_mol = copy.deepcopy(new_mol) # clone first
        for i in range(atom_num):
            x,y,z = pos_noise_coords[i]
            coord_conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        mol = new_mol
        
        rotate_dihedral_label_lst = []
    elif add_noise_type == 0:
    
        if add_ring_noise:
            ring_mask = []
            for atom in mol.GetAtoms(): # j
                if atom.IsInRing():
                    ring_mask.append(True)
                else:
                    ring_mask.append(False)
            noise = torch.randn_like(torch.tensor(pos_coords)) * coord_var
            data_noise = pos_coords + noise.numpy()
            # set back to conf
            for i in range(atom_num):
                if ring_mask[i]: # only add ring noise
                    x,y,z = data_noise[i]
                    conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
            

        # get ring info
        ring_array = []
        for ring in mol.GetRingInfo().AtomRings():
            ring_array.append(ring)


        for bond in mol.GetBonds():
            i_idx = bond.GetBeginAtomIdx()
            j_idx = bond.GetEndAtomIdx()
            # if mol.GetAtomWithIdx(i_idx).IsInRing() and mol.GetAtomWithIdx(j_idx).IsInRing():
            #     continue
            if check_in_samering(i_idx, j_idx, ring_array):
                continue

            org_bond_len = GetBondLength(conf, [i_idx, j_idx])
            
            if mol_param is not None:
                try:
                    bond_para = mol_param[0]["Bonds"][(i_idx, j_idx)].k._magnitude
                    bond_item_var = np.sqrt(1 / bond_para)
                except Exception as e:
                    # print(f'Exception {e}, No bond param for key {ky}, bond: {i_idx}_{j_idx}')
                    bond_item_var = bond_var
            else:
                bond_item_var = bond_var
            # add gaussian noise:
            noise_bond_len = np.random.normal(loc=org_bond_len, scale=bond_item_var)
            # set bond_length
            SetBondLength(conf, [i_idx, j_idx], noise_bond_len)
            
        # angle noise
        for atom in mol.GetAtoms(): # j
            j_idx = atom.GetIdx()
            # atom_symbol = atom.GetSymbol()
            atom_degree = atom.GetDegree()
            if atom_degree >= 2:
                # get neighbors
                neighbors = atom.GetNeighbors()
                neb_lst = []
                for neb in neighbors:
                    neb_lst.append(neb.GetIdx())

                if mol.GetAtomWithIdx(j_idx).IsInRing(): # if j in ring, must pick one neb in ring as i
                    for n_idx in neb_lst:
                        if check_in_samering(j_idx, n_idx, ring_array):
                            i_idx = n_idx
                            break
                else:
                    # j not in ring, random pick one as i
                    i_idx = random.choice(neb_lst)
                
                neb_lst.remove(i_idx)
                # iterate k
                for k_idx in neb_lst:
                    # judge (i, j) and (j, k) in ring:
                    if check_in_samering(i_idx, j_idx, ring_array) and check_in_samering(j_idx, k_idx, ring_array):
                        continue
                    # add angle noise to (i, j, k)
                    org_angle = GetAngle(conf, [i_idx, j_idx, k_idx])
                    if math.isnan(org_angle): # may be nan
                        continue
                    # add noise
                    if mol_param is not None:
                        try:
                            angle_para = mol_param[0]["Angles"][(i_idx, j_idx, k_idx)].k._magnitude
                            angle_item_var = np.sqrt(1 / angle_para)
                        except Exception as e:
                            # print(f'Exception {e}, No angle param for key {ky}, angle: {i_idx}_{j_idx}_{k_idx}')
                            angle_item_var = angle_var
                    else:
                        angle_item_var = angle_var
                    # FIX by change the unit to radian from degree
                    noise = np.random.normal(scale=angle_item_var)
                    noise_angle = org_angle + noise # np.rad2deg(noise)
                    # * 57.3
                    if noise_angle >= 180:
                        noise_angle = 360 - noise_angle # cut the influence to the dih
                    elif noise_angle <= 0:
                        continue
                    
                    # valid_value = wiki_dihedral(conf, [14, 0, 3, 5])
                    # print(f"before add angel noise, [14, 0, 3, 5] dihedral is {valid_value}")
                    
                    # noise_angle = np.random.normal(loc=org_angle, scale=angle_var)
                    SetAngle(conf, [i_idx, j_idx, k_idx], noise_angle)
                    
                    # valid_value = wiki_dihedral(conf, [14, 0, 3, 5])
                    # print(f"after add angel noise, [14, 0, 3, 5] dihedral is {valid_value}, noise is {noise}")
                    # if valid_value > 175:
                    #     print('debug')
        
        # dihedral angle(rotatable or not) [i, j, k, l]
        # get the all the rotatable angel idx
        rotable_bonds = get_torsions([mol]) # format like [(0, 5, 10, 7), (1, 6, 12, 11), (6, 12, 11, 4)]
        
        rotable_sets = set([])
        for rb in rotable_bonds:
            rotable_sets.add(f'{rb[1]}_{rb[2]}')
            rotable_sets.add(f'{rb[2]}_{rb[1]}')

        # dihedral_label_lst = [] # [i, j, k, l, delta_angle]
        rotate_dihedral_label_lst = []
        for bond in mol.GetBonds():

            is_rotate = False

            j_idx = bond.GetBeginAtomIdx()
            k_idx = bond.GetEndAtomIdx()
            # check (j_idx, k_idx) in ring or not
            if check_in_samering(j_idx, k_idx, ring_array):
                continue

            
            j_atom = mol.GetAtomWithIdx(j_idx)
            j_atom_degree = j_atom.GetDegree()
            k_atom = mol.GetAtomWithIdx(k_idx)
            k_atom_degree = k_atom.GetDegree()

            if j_atom_degree < 2 or k_atom_degree < 2: # cannot compose a dihedral angle
                continue

            # get neibors
            j_neighbors = j_atom.GetNeighbors()
            j_neb_lst = []
            for neb in j_neighbors:
                j_neb_lst.append(neb.GetIdx())
            j_neb_lst.remove(k_idx)

            k_neighbors = k_atom.GetNeighbors()
            k_neb_lst = []
            for neb in k_neighbors:
                k_neb_lst.append(neb.GetIdx())
            k_neb_lst.remove(j_idx)

            # random pick one neighbor from j and k, taken as i, l
            i_idx = random.choice(j_neb_lst)
            l_idx = random.choice(k_neb_lst)

            if f'{j_idx}_{k_idx}' in rotable_sets: # rotatable
                deh_var = torsion_var_r
                is_rotate = True
            else:
                deh_var = torsion_var
                is_rotate = False
            
            if mol_param is not None:
                try:
                    torsion_para = mol_param[0]["ProperTorsions"][(i_idx, j_idx, k_idx, l_idx)].k[0]._magnitude 
                    torsion_period = mol_param[0]["ProperTorsions"][(i_idx, j_idx, k_idx, l_idx)].periodicity[0]  #for torsion i-j-k-l
                    
                    if torsion_para == 0:
                        continue
                    
                    sqrt_value = 1 / (torsion_para*(torsion_period**2))
                    if sqrt_value < 0:
                        continue
                                      
                    deh_var_item = np.sqrt(sqrt_value)
                except Exception as e:
                    deh_var_item = deh_var
                    # print(f'Exception {e}, No torsion param for key {ky}, torsion: {i_idx}_{j_idx}_{k_idx}_{l_idx}')
                    
                
            else:
                deh_var_item = deh_var
            # torsion_para = para[0]["ProperTorsions"][(i_idx, j_idx, k_idx, l_idx)].k[0]._value 
# torsion_period=para[0]["ProperTorsions"][(i_idx, j_idx, k_idx, l_idx)].periodicity[0]  #for torsion i-j-k-l
            
            
            # FIX by change the unit to radian from degree
            noise = np.random.normal(scale=deh_var_item)
            

            # noise_deh_angle = np.random.normal(loc=org_deh_angle, scale=deh_var)
            for l_idx in k_neb_lst:
                # add noise
                org_deh_angle = GetDihedral(conf, [i_idx, j_idx, k_idx, l_idx])
                if math.isnan(org_deh_angle): # may be nan
                    continue
                noise_deh_angle = org_deh_angle + noise # np.rad2deg(noise)
                # valid_value = wiki_dihedral(conf, [14, 0, 3, 5])
                # print(f"before add noise, [14, 0, 3, 5] dihedral is {valid_value}")
                SetDihedral(conf, [i_idx, j_idx, k_idx, l_idx], noise_deh_angle)
                # valid_value = wiki_dihedral(conf, [14, 0, 3, 5])
                # print(f"after add noise, [14, 0, 3, 5] dihedral is {valid_value}, noise is {noise}")
                # if valid_value > 175:
                #     print('debug')

        #     if is_rotate:
        #         rotate_dihedral_label_lst.append([i_idx, j_idx, k_idx, l_idx, noise_deh_angle - org_deh_angle])
        #     else:
        #         dihedral_label_lst.append([i_idx, j_idx, k_idx, l_idx, noise_deh_angle - org_deh_angle])
        
        # get the difference between mol and opt_mol
    edge_idx, angle_idx, dihedral_idx = get_2d_gem(opt_mol)

    bond_len_lst, angle_lst, dihedral_lst = get_info_by_gem_idx(opt_mol, edge_idx, angle_idx, dihedral_idx)
    bond_len_lst_noise, angle_lst_noise, dihedral_lst_noise = get_info_by_gem_idx(mol, edge_idx, angle_idx, dihedral_idx)
    # noise difference, has filter nan
    bond_label_lst, angle_label_lst, dihedral_label_lst = concat_idx_label(edge_idx, angle_idx, dihedral_idx, bond_len_lst_noise-bond_len_lst, angle_lst_noise-angle_lst, dihedral_lst_noise-dihedral_lst)
    # edge: [i,j, noise(i,j)]
    # angle: [i, j, k, noise(i,j,k)]
    # filter zero
    if angle_label_lst.size != 0:
        angle_label_lst = angle_label_lst[angle_label_lst[:,-1] != 0]

    # filter zero
    if dihedral_label_lst.size != 0:
        dihedral_label_lst = dihedral_label_lst[dihedral_label_lst[:, -1] != 0]

    if add_noise_type == 1:
        mol = [mol, opt_mol]
    
    
    if len(bond_label_lst): # denan
        bond_label_lst = np.array(bond_label_lst, dtype=np.float32)
        # bond_label_lst[:,2] = noise_bond - org_bond
        mask = ~np.isnan(bond_label_lst[:, 2])
        bond_label_lst = bond_label_lst[mask]
        
    if len(angle_label_lst): # denan
        angle_label_lst = np.array(angle_label_lst, dtype=np.float32)
        # angle_label_lst[:, 3] = noise_angle - org_angle
        mask = ~np.isnan(angle_label_lst[:, 3])
        angle_label_lst = angle_label_lst[mask]
        
    
    specific_var_lst = []
    if mol_param is not None:
        for bond_label in bond_label_lst:
            i_idx, j_idx = int(bond_label[0]), int(bond_label[1])
            try:
                bond_para = mol_param[0]["Bonds"][(i_idx, j_idx)].k._magnitude
                bond_item_var = np.sqrt(1 / bond_para)
            except Exception as e:
                # print(f'Exception {e}, No bond param for key {ky}, bond: {i_idx}_{j_idx}')
                bond_item_var = bond_var
            specific_var_lst.append(bond_item_var)
        for angle_label in angle_label_lst:
            i_idx, j_idx, k_idx = int(angle_label[0]), int(angle_label[1]), int(angle_label[2])
            try:
                angle_para = mol_param[0]["Angles"][(i_idx, j_idx, k_idx)].k._magnitude
                angle_item_var = np.sqrt(1 / angle_para)
            except Exception as e:
                # print(f'Exception {e}, No angle param for key {ky}, angle: {i_idx}_{j_idx}_{k_idx}')
                angle_item_var = angle_var
            specific_var_lst.append(angle_item_var)
        
        for torsion_label in dihedral_label_lst:
            i_idx, j_idx, k_idx, l_idx = int(torsion_label[0]), int(torsion_label[1]), int(torsion_label[2]), int(torsion_label[3]),
            
            if f'{j_idx}_{k_idx}' in rotable_sets: # rotatable
                deh_var = torsion_var_r
                is_rotate = True
            else:
                deh_var = torsion_var
                is_rotate = False
            
            try:
                torsion_para = mol_param[0]["ProperTorsions"][(i_idx, j_idx, k_idx, l_idx)].k[0]._magnitude 
                torsion_period = mol_param[0]["ProperTorsions"][(i_idx, j_idx, k_idx, l_idx)].periodicity[0]
                if torsion_para == 0:
                    deh_var_item = deh_var
                else:
                    sqrt_value = 1 / (torsion_para*(torsion_period**2))
                    if sqrt_value < 0:
                        deh_var_item = deh_var
                    else:                
                        deh_var_item = np.sqrt(sqrt_value)
            except Exception as e:
                deh_var_item = deh_var
            
            specific_var_lst.append(deh_var_item)
    
    
    
    return mol, bond_label_lst, angle_label_lst, dihedral_label_lst, rotate_dihedral_label_lst, specific_var_lst
    # return mol, bond_label_lst, angle_label_lst, dihedral_label_lst, rotate_dihedral_label_lst



if __name__ == "__main__":
    
    import pickle
    import lmdb
    from tqdm import tqdm
    
    # test use lmdb param
    MOL_LST = lmdb.open('/data/protein/SKData/DenoisingData/MOL_LMDB', readonly=True, subdir=True, lock=False)
    Param_Lst = lmdb.open('/data/protein/SKData/DenoisingData/Param_DB', readonly=True, subdir=True, lock=False)
    ky = str('1').encode()
    

    # get all key of param 
    
    with Param_Lst.begin() as txn:
        _keys = list(txn.cursor().iternext(values=False))
    
    param_idx = []
    for ky in _keys:
        param_idx.append(int(ky))
    
    np.save('param_idx.npy', param_idx)
    
    for ky in tqdm(_keys):
        serialized_data = MOL_LST.begin().get(ky)
        mol = pickle.loads(serialized_data)   
        serialized_data = Param_Lst.begin().get(ky)
        mol_param = pickle.loads(serialized_data)
        add_equi_noise_new(mol, mol_param=mol_param, ky=ky)
    


    
    with open('org_mol_890.pkl', 'rb') as fr:
        org_mol = pickle.load(fr)
    org_conf = org_mol.GetConformer()
    for i in range(100):
        noise_mol, bond_label_lst, angle_label_lst, dihedral_label_lst, rotate_dihedral_label_lst = add_equi_noise_new(org_mol)
        noise_conf = noise_mol.GetConformer()
        print(wiki_dihedral(noise_conf, [14, 0, 3, 5]))
    exit(0)
    
    from tqdm import tqdm
    mol = Chem.SDMolSupplier('org_0.sdf')[0]
    # rotate_bonds = get_torsions([mol])
    # print(rotate_bonds)
    # print(get_rotate_order_info(mol, rotate_bonds))
    # dm = np.load('dihmol.npy', allow_pickle=True)[0]
    # noise_mol, bond_label_lst, angle_label_lst, dihedral_label_lst = add_equi_noise(mol, add_ring_noise=True)
    MOL_LST = np.load("/home/fengshikun/DenoisingData/mols_10w_atomLess30.npy", allow_pickle=True)
    # MOL_LST = np.load('coord_noise_nan.npy', allow_pickle=True)
    test_mol = MOL_LST[0]
    # from torsion_utils import add_equi_keep_noise
    # mol, bond_label_lst, angle_label_lst, r_dihedral_label_lst, r_rotate_dihedral_label_lst = add_equi_keep_noise2(test_mol, add_ring_noise=True, angle_var=0.09) # angle_var=0.09
    # mol, bond_label_lst, angle_label_lst, r_dihedral_label_lst, r_rotate_dihedral_label_lst = add_equi_keep_noise2(test_mol, add_ring_noise=True, angle_var=0.09) # angle_var=0.09

    # mol, bond_label_lst, angle_label_lst, r_dihedral_label_lst, r_rotate_dihedral_label_lst = add_equi_keep_noise2(test_mol, add_ring_noise=True, angle_var=0.09) # angle_var=0.09
    mol, bond_label_lst, angle_label_lst, dihedral_label_lst, rotate_dihedral_label_lst = add_equi_noise_new(test_mol, add_ring_noise=True) # angle_var=0.09

    from torchmdnet.models.feats import angle_emb, dist_emb
    dt_emb = dist_emb(num_radial=6)
    
    org_conf = test_mol.GetConformer()
    noise_conf = mol.GetConformer()
    org_bond_lst = []
    for ele in bond_label_lst:
        org_bond_lst.append(GetBondLength(org_conf, [int(ele_i) for ele_i in ele[:2]]))
    bond_noise = torch.tensor(bond_label_lst[:,-1])# to radian
    org_bond_lst = torch.tensor(org_bond_lst)
    org_bond_emb = dt_emb(org_bond_lst)
    noise_bond_emb = dt_emb(org_bond_lst + bond_noise)


    ang_emb = angle_emb(num_spherical=3, num_radial=6)
    org_angle_lst = []
    noise_angle_lst = []
    org_new_angle_lst = []

    org_torsion_lst = []
    org_torsion_new_lst = []
    noise_torsion_lst = []
    noise_torsion_new_lst = []
    tor_noise_lst = dihedral_label_lst[:,-1]

    for ele in angle_label_lst:
        org_angle_lst.append(GetAngle(org_conf, [int(ele_i) for ele_i in ele[:3]]))
        org_new_angle_lst.append(getAngle_new(org_conf, [int(ele_i) for ele_i in ele[:3]]))
        noise_angle_lst.append(GetAngle(noise_conf, [int(ele_i) for ele_i in ele[:3]]))

    for ele in dihedral_label_lst:
        org_torsion_lst.append(GetDihedral(org_conf, [int(ele_i) for ele_i in ele[:4]]))
        org_torsion_new_lst.append(wiki_dihedral(org_conf, [int(ele_i) for ele_i in ele[:4]]))
        noise_torsion_lst.append(GetDihedral(noise_conf, [int(ele_i) for ele_i in ele[:4]]))
        noise_torsion_new_lst.append(wiki_dihedral(noise_conf, [int(ele_i) for ele_i in ele[:4]]))

        
    # get coordinate 
    atom_num = mol.GetNumAtoms()
    coords = np.zeros((atom_num, 3), dtype=np.float32)
    org_coords = np.zeros((atom_num, 3), dtype=np.float32)
    for idx in range(atom_num):
        c_pos = noise_conf.GetAtomPosition(idx)
        coords[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]
        n_pos = org_conf.GetAtomPosition(idx)
        org_coords[idx] = [float(n_pos.x), float(n_pos.y), float(n_pos.z)]
    
    coords = torch.tensor(coords)
    org_coords = torch.tensor(org_coords)
    t_idx = torch.tensor(dihedral_label_lst[:,:4], dtype=torch.long)
    # i_pos, j_pos, k_pos, l_pos = t_idx[0], t_idx[1], t_idx[2], t_idx[3]
    torch_org_torsion = wiki_dihedral_torch(org_coords, t_idx)
    torch_noise_torsion = wiki_dihedral_torch(coords, t_idx)

    a_idx = torch.tensor(angle_label_lst[:, :3], dtype=torch.long)
    torch_org_angle = getAngle_torch(org_coords, a_idx)
    torch_noise_angle = getAngle_torch(coords, a_idx)

    
    # get coordinate
    atom_num = mol.GetNumAtoms()
    origin_pos = np.zeros((atom_num, 3), dtype=np.float32)
    noise_pos = np.zeros((atom_num, 3), dtype=np.float32)
    # pos_noise_coords = new_mol.GetConformer().GetPositions()
    for idx in range(atom_num):
        c_pos = org_conf.GetAtomPosition(idx)
        origin_pos[idx] = [float(c_pos.x), float(c_pos.y), float(c_pos.z)]

        n_pos = noise_conf.GetAtomPosition(idx)
        noise_pos[idx] = [float(n_pos.x), float(n_pos.y), float(n_pos.z)]
    


    angle_noise = torch.tensor(angle_label_lst[:,-1]) / 57.3 # to radian
    org_angle_lst = torch.tensor(org_angle_lst) / 57.3 # to radian
    org_emb = ang_emb.forward_angle(org_angle_lst)
    change_emb = ang_emb.forward_angle(org_angle_lst + angle_noise)

    angle_test_emb = ang_emb.forward_angle(torch.tensor([-1, 1], dtype=torch.float))
    print(angle_label_lst)
    # for mol in tqdm(MOL_LST):
    #     noise_mol, bond_label_lst, angle_label_lst, dihedral_label_lst = add_equi_noise(mol)
    #     dihedral_label = np.array(dihedral_label_lst)
    #     if np.isnan(dihedral_label).sum():
    #         import pdb; pdb.set_trace()
    #         print('nan happens!')
