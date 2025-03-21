import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import numpy as np

# 原子坐标文件内容
atom_coordinates = """
C 0.117376335 -0.734439135 -2.332915545
H -1.834425330 1.807372093 2.271738529
C -1.168139577 0.462256223 0.430944413
O 1.216177702 -2.374994993 -1.232559800
O 0.308235675 3.029060602 2.101994038
H 2.896190643 -1.128077507 -1.323209286
C -0.391547620 0.507742941 1.643912077
H 2.846302032 -2.660824537 -1.896409988
N 2.379848003 -1.758912086 -1.922266483
H -3.052817106 0.870583236 1.705348969
H -1.716917753 0.589329898 -0.497953832
C 0.195188805 0.437600046 0.284153908
H 0.379905462 0.025667667 -3.020589352
H -2.082547903 -0.172651365 2.540101528
N 0.949465156 0.161767438 -0.860475898
H -0.630389988 -1.439919829 -2.671631575
C -1.959057450 0.747728229 1.905124426
H 0.649997473 2.414264917 1.551533818
C 0.744105101 -1.139101148 -1.237060308
H 0.153050467 0.355547488 2.560220242
"""

# 解析坐标文件
lines = atom_coordinates.strip().split("\n")
atoms = []
coords = []

for line in lines:
    parts = line.split()
    atoms.append(parts[0])
    coords.append([float(x) for x in parts[1:]])

# 创建RDKit分子对象
mol = Chem.RWMol()
for atom in atoms:
    a = Chem.Atom(atom)
    mol.AddAtom(a)

# 设置原子坐标
conf = Chem.Conformer(len(atoms))
for i, coord in enumerate(coords):
    conf.SetAtomPosition(i, coord)

mol.AddConformer(conf)

# 生成分子图
AllChem.Compute2DCoords(mol)
img = Draw.MolToImage(mol)
# img.show()
img.save("rdkit_molecule.png")
