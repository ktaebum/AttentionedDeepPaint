import os
import numpy as np
import matplotlib.pyplot as plt

log_root = './data/logs/attention/'

log_file = os.path.join(log_root, 'loss.txt')

datas = np.loadtxt(log_file)

d_loss_real = datas[:, 0]
d_loss_fake = datas[:, 1]
g_loss = datas[:, 2]
re_loss = datas[:, 3]
epochs = range(1, len(re_loss) + 1)

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.plot(
    epochs,
    d_loss_real,
    marker='o',
    label='D Loss Real',
    linewidth=3,
    markersize=8)
ax.plot(
    epochs,
    d_loss_fake,
    marker='o',
    label='D Loss Fake',
    linewidth=3,
    markersize=8)

ax.plot(epochs, g_loss, marker='o', label='G Loss', linewidth=3, markersize=8)
ax.plot(
    epochs,
    re_loss / 100,
    marker='o',
    label='Reconstruction Loss',
    linewidth=3,
    markersize=8)

ax.set_xlabel('Epochs', size=18)
ax.set_xticks(epochs)

ax.set_ylabel('Loss Value', size=18)

ax.tick_params(labelsize=15)

plt.legend(fontsize=18)
plt.show()