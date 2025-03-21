import torch


denoise_error_lst = torch.load('denoise_error_new.pt')
denoise_error_lst = denoise_error_lst.T
value_lst = []

for dl in denoise_error_lst:
    value_lst.append(dl.mean().item())

t = [i for i in range(1000)]

import matplotlib.pyplot as plt

# draw the t and value curve
plt.plot(t, value_lst)
plt.savefig('denoise_error.png')

print(value_lst)