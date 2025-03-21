from rdkit import Chem
import numpy as np

def add_gaussian_noise(molecule, noise_level=0.1):
    """
    给分子的坐标添加高斯噪声
    :param molecule: RDKit 分子对象
    :param noise_level: 噪声的标准差
    :return: None, 直接修改分子对象的坐标
    """
    conf = molecule.GetConformer()
    for i in range(molecule.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        noise = np.random.normal(0, noise_level, 3)  # 生成 3D 高斯噪声
        conf.SetAtomPosition(i, pos + noise)

def process_sdf(input_sdf, output_sdf, noise_level=0.1):
    """
    从输入 SDF 文件读取分子，添加噪声，然后写入输出 SDF 文件
    :param input_sdf: 输入 SDF 文件路径
    :param output_sdf: 输出 SDF 文件路径
    :param noise_level: 噪声的标准差
    """
    suppl = Chem.SDMolSupplier(input_sdf, removeHs=False)
    writer = Chem.SDWriter(output_sdf)

    for mol in suppl:
        if mol is not None:
            add_gaussian_noise(mol, noise_level)
            writer.write(mol)

    writer.close()

# 示例用法
input_sdf_file = '/mnt/nfs-ssd/data/fengshikun/e3_diffusion_for_molecules/outputs/split_k8_t10_with_atom_type_prop_pred/epoch_2640_0/chain/test.sdf'  # 替换为你的输入 SDF 文件路径
output_sdf_file = '/mnt/nfs-ssd/data/fengshikun/e3_diffusion_for_molecules/outputs/split_k8_t10_with_atom_type_prop_pred/epoch_2640_0/chain/test1.sdf'  # 替换为你的输出 SDF 文件路径
process_sdf(input_sdf_file, output_sdf_file, noise_level=0.05)