import torch
import time

d_model=512
seq_len=100

pos=torch.arange(start=0, end=seq_len, step=1).unsqueeze(1)

tic=time.time()
two_i=torch.arange(start=0, end=d_model-1, step=2).unsqueeze(0)
theta=pos/(10000**(two_i/d_model))
toc=time.time()
elapsed_time=toc-tic
print(f'elapsed time of atteantion is all you need: {elapsed_time*1000}')


tic=time.time()
div_term=torch.exp(torch.arange(start=0, end=d_model-1, step=2)*(-torch.log(torch.tensor([10000])))/d_model)
theta=pos*div_term
toc=time.time()
elapsed_time=toc-tic
print(f'elapsed time of harvard: {elapsed_time*1000}')
