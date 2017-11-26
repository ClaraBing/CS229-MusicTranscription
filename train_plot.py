import numpy as np
import matplotlib.pyplot as plt

log4 = [float(line.strip()) for line in open('log4', 'r').readlines()]
log5 = [float(line.strip()) for line in open('log5', 'r').readlines()]
log6 = [float(line.strip()) for line in open('log6', 'r').readlines()]

log_files = ['conv5_11_gpu.log', 'conv5_12_gpu.log']
lr = [0.001, 0.001]
momentum = [0.5, 0.9]
for i, log in enumerate(log_files):
	lines = open(log, 'r').readlines()
	loss_lines = [line for line in lines if 'Loss' in line and 'Train' in line]
	losses = [float(line.split(':')[2].split('\t')[0]) for line in loss_lines]
	plt.plot(range(0, len(losses)*10, 10), losses, label='lr={:f}, momentum={:f}'.format(lr[i], momentum[i]))

fig = plt.gcf()
fig.set_size_inches(20, 10.5)
plt.legend()
plt.title('iteration vs Neg LL')

plt.savefig('plot_11_12.png', dpi=300)
plt.clf()