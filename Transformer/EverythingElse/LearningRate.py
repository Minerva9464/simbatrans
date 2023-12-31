import matplotlib.pyplot as plt
import numpy as np

def lr_function(low, up, epoch_number, percentage_to_fall_down, percentage_to_rest, epoch):
    if epoch < epoch_number*percentage_to_fall_down:
        return up
    elif epoch > epoch_number*percentage_to_rest:
        return low
    else:
        # m = (up-low)/(epoch_number*percentage_to_fall_down - epoch_number*percentage_to_rest)
        # b = up - m*epoch_number*percentage_to_fall_down
        # return m*epoch+b
        return 0.5*(1+np.cos(np.pi * epoch/epoch_number))*up+low

epoch_number = 250
epochs = np.array(range(1,epoch_number+1))
lrs = np.array([])
low = 1e-6
up = 1e-4
percentage_to_fall_down = 0.02
percentage_to_rest = 0.9

for epoch in epochs:
    lrs = np.append(lrs, lr_function(
        low, up, epoch_number, percentage_to_fall_down, percentage_to_rest, epoch
    ))
    
plt.plot(epochs, lrs)
plt.savefig('Achievements/Learning Rate.png', dpi=300)
plt.show()
plt.close()

