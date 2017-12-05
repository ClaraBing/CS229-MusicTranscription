import numpy as np
import matplotlib.pyplot as plt

bad_prob = np.load("bad_prob_hmm_refined.npy")
# bad_bin = np.load("bad_bin_hmm.npy")
good_prob = np.load("good_prob_hmm_refined.npy")
# good_bin = np.load("good_bin_hmm.npy")




# Plot most probable probabilities
data = [good_prob[:,0], bad_prob[:,0]]
plt.hist(data, alpha=0.7, label=['good', 'bad'], bins=100)
plt.title("Repartition of 1st probabilities")
plt.legend(loc='upper right')
plt.yscale('log', nonposy='clip')
plt.show()

# Plot 2nd most probable bins
data = [good_prob[:,1], bad_prob[:,1]]
plt.hist(data, alpha=0.7, label=['good', 'bad'], bins=100)
plt.title("Repartition of 2nd probabiltiies")
plt.legend(loc='upper right')
plt.yscale('log', nonposy='clip')
plt.show()


# # Plot most probable bins
# data = [good_bin[:,0], bad_bin[:,0]]
# plt.hist(data, alpha=0.7, label=['good', 'bad'], bins=109)
# plt.title("Repartition of 1st bins")
# plt.legend(loc='upper right')
# plt.yscale('log', nonposy='clip')
# plt.show()
#
# # Plot 2nd most probable bins
# data = [good_bin[:,1], bad_bin[:,1]]
# plt.hist(data, alpha=0.7, label=['good', 'bad'], bins=109)
# plt.title("Repartition of 2nd bins")
# plt.legend(loc='upper right')
# plt.yscale('log', nonposy='clip')
# plt.show()
