# read the file content like this:
# t is 1.0, sigma: 0.5340599417686462
# draw the curve of t and simga with matplotlib

import matplotlib.pyplot as plt
import numpy as np
import re

t = []
sigma = []
z_coeff = []
nn_coeff = []

pattern_t = r"t is ([\d.]+)"
pattern_sigma = r"sigma: ([\d.]+)"
pattern_z = r"z coeffient: ([\d.]+)"
pattern_nn = r"nn output coeffient: ([\d.]+)"

with open('/mnt/nfs-ssd/data/fengshikun/e3_diffusion_for_molecules/equivariant_diffusion/sigma_list.txt') as f:
    for line in f:
        if 't is' in line:
            # t.append(float(line.split(',')[0].split(' ')[-1]))
            # sigma.append(float(line.split(':')[-1]))
            # parse t, sigma ,z_coeff, nn_coeff from line 
            # t is 1.0, sigma: 0.5340599417686462, z coeffient: 1.182807207107544, nn output coeffient: 0.3373633027076721
            # t.append(float(line.split(',')[0].split(' ')[-1]))
            # sigma.append(float(line.split(':')[-3]))
            # z_coeff.append(float(line.split(':')[-2]))
            # nn_coeff.append(float(line.split(':')[-1]))
            t_match = re.findall(pattern_t, line)
            sigma_match = re.findall(pattern_sigma, line)
            z_match = re.findall(pattern_z, line)
            nn_match = re.findall(pattern_nn, line)

            # Print the extracted values
            if t_match:
                t_v = float(t_match[0])

            if sigma_match:
                sigma_v = float(sigma_match[0])

            if z_match:
                z = float(z_match[0])

            if nn_match:
                nn = float(nn_match[0])
            
            t.append(t_v)
            sigma.append(sigma_v)
            z_coeff.append(z)
            nn_coeff.append(nn)
            
                        
plt.figure(figsize=(8, 6))  # Optional: adjust figure size
plt.plot(t, sigma, marker='o', linestyle='-', color='b', label='sigma', linewidth=0.5, markersize=1)
# t and z_coeff
plt.plot(t, z_coeff, marker='o', linestyle='-', color='r', label='z_coeff', linewidth=0.5, markersize=1)
# t and nn_coeff
plt.plot(t, nn_coeff, marker='o', linestyle='-', color='g', label='nn_coeff', linewidth=0.5, markersize=1)

gamma = np.square(np.array(sigma)) / 10

# new_gamma = gamma.clone()



plt.plot(t, gamma, marker='o', linestyle='-', color='c', label='gamma', linewidth=0.5, markersize=1)

plt.plot(t, gamma/0.04, marker='o', linestyle='-', color='m', label='gamma/0.04', linewidth=0.5, markersize=1)

plt.title('Plot of t vs sigma')
plt.xlabel('t values')
plt.ylabel('sigma values')
plt.grid(True)
plt.legend()

plt.savefig('draw_sigma.png')