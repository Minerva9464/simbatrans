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
        return (1+np.cos(np.pi * epoch/epoch_number))*up+low

epoch_number = 500
epochs = np.array(range(1,epoch_number+1))
lrs = np.array([])
low = 1e-4
up = 1e-2
percentage_to_fall_down = 0.10
percentage_to_rest = 0.70

for epoch in range(epoch_number):
    lrs = np.append(lrs, lr_function(
        low, up, epoch_number, percentage_to_fall_down, percentage_to_rest, epoch
    ))
    
plt.plot(epochs, lrs)
plt.show()

