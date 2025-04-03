# UniGEM: A Unified Approach to Generation and Property Prediction for Molecules (ICLR 25)

This is the official implementation of the paper **UniGEM: A Unified Approach to Generation and Property Prediction for Molecules (ICLR 25).**

---

## Download

### Pretrained Models

Download link: [Baidu Netdisk](https://pan.baidu.com/s/1MFs_AOdmMA1KiYYW7mPlvA)  
Extract code: **j6ih**  

Download link: [DropBox](https://www.dropbox.com/scl/fo/l7apifqyp9bgdu7flrlrr/APisremaZ5XfievFK6wnFQI?rlkey=ulfrhizja7qptxn7sv0s3lb9c&st=13v8cs6o&dl=0)

- **split_k8_t10_with_atom_type_prop_pred_lumo**: Unified QM9 model for molecular generation and property prediction (LUMO).  
- **geom_drugs_model**: Model for **GEOM-Drugs** molecular generation.  
- **bfn_k8_atomtype_prop_pred_t150_resume3_resume**: QM9 generation model with **BFN** coordinate generation algorithm.  

### Training Data (TODO)

You can refer to the **EDM repo** for training data, or wait for us to update the processed dataset.

---

## Training and Testing

### QM9

#### Train UniGEM with Property Prediction (LUMO), Nucleation Time = 10  

```bash
CUDA_VISIBLE_DEVICES=0 python -u main_qm9.py --n_epochs 3000 --exp_name split_k8_t100_with_atom_type_prop_pred \
    --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 \
    --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 \
    --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --property_pred 1 --prediction_threshold_t 10 \
    --model DGAP --sep_noisy_node 1 --target_property lumo --atom_type_pred 1 --branch_layers_num 8 \
    --use_prop_pred 1 > split_k8_t100_with_atom_type_prop_pred.log 2>&1 &
```

#### Test Model - Generation

```bash
python -u eval_analyze.py --model_path /nfs/SKData/ssd_data/UniGEM_Data/models/split_k8_t10_with_atom_type_prop_pred_homo \
    --n_samples 10_000 --save_to_xyz 1 --checkpoint_epoch 2000
```


#### Test Model - Property Prediction

```bash
cd qm9/property_prediction
CUDA_VISIBLE_DEVICES=0 python -u eval_prop_pred.py --num_workers 2 --lr 5e-4 --property lumo --model_name egnn \
    --generators_path /nfs/SKData/ssd_data/UniGEM_Data/models/split_k8_t10_with_atom_type_prop_pred_homo \
    --model_path generative_model_ema_2000.npy
```

#### Training on GEOM-Drugs (Requires 4 GPUs)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main_geom_drugs.py --n_epochs 3000 --exp_name geom_drugs_k3_atom_type_pred_nf1 \
    --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 \
    --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 32 --nf 256 --n_layers 4 --lr 1e-4 \
    --normalize_factors [1,4,10] --test_epochs 1 --ema_decay 0.9999 --prediction_threshold_t 10 --model DGAP \
    --sep_noisy_node 1 --target_property lumo --atom_type_pred 1 --branch_layers_num 3 --normalization_factor 1
```

####  Test on GEOM-Drugs

```bash
CUDA_VISIBLE_DEVICES=2 python -u eval_analyze.py --model_path outputs/geom_drugs_k3_atom_type_pred_nf1 \
    --n_samples 10_000 --save_to_xyz 0 --checkpoint_epoch 13 > geom_drugs_k3_atom_type_pred_nf1_gen_epoch13.log 2>&1 &
```

### Adapting UniGEM to BFN Generation Algorithm
UniGEM can also be adapted to more powerful generation algorithms like BFN.

#### Train UniGEM with BFN (Nucleation Time = 150)

```bash
CUDA_VISIBLE_DEVICES=7 python -u main_qm9.py --n_epochs 3000 --exp_name bfn_k8_atomtype_prop_pred_t150 \
    --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 \
    --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 \
    --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --model DGAP --target_property homo \
    --sep_noisy_node 1 --num_workers 4 --bfn_schedule 1 --atom_type_pred 1 --branch_layers_num 8 \
    --use_prop_pred 1 --property_pred 1 --prediction_threshold_t 150
```

#### Test BFN Model

```bash
CUDA_VISIBLE_DEVICES=7 python -u eval_analyze.py --model_path /nfs/SKData/ssd_data/UniGEM_Data/models/bfn_k8_atomtype_prop_pred_t150_resume3_resume \
    --n_samples 100 --save_to_xyz 0 --checkpoint_epoch 2980 --sample_steps 1000
```
The codebase is modified based on EDM: [e3_diffusion_for_molecules](https://github.com/ehoogeboom/e3_diffusion_for_molecules).




### Cite  
If you find our work or code useful, please consider citing our paper:  

```bibtex
@article{feng2024unigem,
  title={UniGEM: A Unified Approach to Generation and Property Prediction for Molecules},
  author={Feng, Shikun and Ni, Yuyan and Lu, Yan and Ma, Zhi-Ming and Ma, Wei-Ying and Lan, Yanyan},
  journal={arXiv preprint arXiv:2410.10516},
  year={2024}
}
```


