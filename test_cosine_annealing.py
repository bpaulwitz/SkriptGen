import matplotlib.pyplot as plt
import math

plt.figure()

epochs = 2
training_len = 500

annealing_rate = 250
max_lr = 3e-4
min_lr = 1e-6

max_global_step = epochs * training_len

x_values = range(max_global_step)
y_values_cos = []
y_values_orig = []
y_values_log = []
upper_lr = max_lr - min_lr
for x in x_values:
    y_values_cos.append(upper_lr / 2.0 * (math.cos(1.0 / annealing_rate * math.pi * (x % annealing_rate)) + 1) + min_lr)
    #                |                                      cosine annealing                                 |                 exponential decrease
    #y_values_orig.append(upper_lr / 2.0 * (math.cos(1.0 / annealing_rate * math.pi * (x % annealing_rate)) + 1) * (-math.exp((math.log(2) / max_global_step) * x) + 2) + min_lr)
    y_values_orig.append(min_lr + 1 / 2 * (max_lr - min_lr) * (1 + math.cos(math.pi * x / annealing_rate)))
    #                |                                      cosine annealing                                 |                 logarithmic decrease
    #y_values_log.append(upper_lr / 2.0 * (math.cos(1.0 / annealing_rate * math.pi * (x % annealing_rate)) + 1) * (-math.log((math.e - 1) / max_global_step * x + 1) + 1) + min_lr)

f, (ax1, ax2) = plt.subplots(2)
ax1.plot(x_values, y_values_orig)
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Learning Rate")
ax1.grid(True)
ax1.set_title("Cosine Annealing with warm restart")
ax2.plot(x_values, y_values_cos)
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Learning Rate")
ax2.grid(True)
ax2.set_title("Cosine Annealing with cold restart")
plt.show()