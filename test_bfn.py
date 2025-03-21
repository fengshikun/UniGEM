import os

base_cmd = "CUDA_VISIBLE_DEVICES=7 python -u eval_analyze.py --model_path /mnt/nfs-ssd/data/fengshikun/e3_diffusion_for_molecules/outputs/{} --n_samples 10_000 --save_to_xyz 0 --checkpoint_epoch {} --sample_steps 1000 > {}_epoch{}_sample1k_fixbug.log 2>&1 &"



test_lst = {
    'bfn_k8_atomtype_prop_pred_t150_resume3_resume': 2800,
    'bfn_k8_atomtype_prop_pred_t150_resume3_resume': 2900,
    'bfn_k8_atomtype_prop_pred_t150_resume3_resume': 2980,
    'bfn_k8_atomtype_prop_pred_t150_resume4_resume': 3100,
    'bfn_k8_atomtype_prop_pred_t150_resume5_resume': 3200,
    'bfn_k8_atomtype_prop_pred_t150_resume5_resume': 3300,
    'bfn_k8_atomtype_prop_pred_t150_resume5_resume': 3400,
}


test_lst = [
    ['bfn_k8_atomtype_prop_pred_t150_resume3_resume', 2800],
    ['bfn_k8_atomtype_prop_pred_t150_resume3_resume', 2900],
    ['bfn_k8_atomtype_prop_pred_t150_resume5_resume', 3200],
    ['bfn_k8_atomtype_prop_pred_t150_resume5_resume', 3300],
]


test_lst = [
    [4, 'bfn_k8_atomtype_prop_pred_t150_resume3_resume', 3700],
    [5, 'bfn_k8_atomtype_prop_pred_t150_resume3_resume', 3800],
    [6, 'bfn_k8_atomtype_prop_pred_t150_resume5_resume', 3900],
    [7, 'bfn_k8_atomtype_prop_pred_t150_resume5_resume', 4000],
]


test_lst = [
    [4, 'bfn_k8_atomtype_prop_pred_t150_resume5_resume', 3900],
    [5, 'bfn_k8_atomtype_prop_pred_t150_resume5_resume', 3900],
    [6, 'bfn_k8_atomtype_prop_pred_t150_resume5_resume', 3900],
    [7, 'bfn_k8_atomtype_prop_pred_t150_resume5_resume', 4180],
]

base_cmd = "CUDA_VISIBLE_DEVICES={} python -u eval_analyze.py --model_path /mnt/nfs-ssd/data/fengshikun/e3_diffusion_for_molecules/outputs/{} --n_samples 10_000 --save_to_xyz 0 --checkpoint_epoch {} --sample_steps 1000 > {}_epoch{}_sample1k_fixbug.log 2>&1 &"

# for key, value in test_lst.items():
#     os.system(base_cmd.format(key, value, key, value))
#     print(base_cmd.format(key, value, key, value))
#     print('Done with {}'.format(key))
for c_id, key, value in test_lst:
    # os.system(base_cmd.format(c_id, key, value, key, value))
    print(base_cmd.format(c_id, key, value, key, value))
    # print('Done with {}'.format(key))