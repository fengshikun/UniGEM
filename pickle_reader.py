import pickle

# 定义pickle文件的路径
file_path = 'outputs/edm_qm9_DGAP_resume/args.pickle'
# file_path = "qm9/property_prediction/outputs/exp_class_lumo/args.pickle"
# file_path = "qm9/temp/qm9_second_half_smiles.pickle"

# 读取pickle文件
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# 打印读取的数据
print(data)
