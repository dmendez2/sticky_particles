import matplotlib.pyplot as plt
import numpy as np

# Reverse engineered log-normal distribution parameters for Mason's diameter measurements of the liquid metal particles
# Mode = exp[mu - sigma^2] -> log(mode) = mu - sigma^2
# Mean = exp[mu + 0.5 * sigma^2] -> 2*log(mean) = 2*mu + sigma^2
# So I can solve for mu = [2*log(mean) + log(mode)]/3
# Then  for sigma = sqrt[mu - log(Mode)]
# OR sigma = sqrt(2* [log(Mean) - mu]) 
def log_normal_parameters(my_mode = 75, my_mean = 248, normalizing_diameter = 248):
    mode = my_mode / normalizing_diameter
    mean = my_mean / normalizing_diameter

    mu = (2*np.log(mean) + np.log(mode))/3
    sigma = np.sqrt(mu - np.log(mode))
    #sigma = np.sqrt(2 * (np.log(mean) - mu))

    return mu, sigma

# Sample particle sizes from exponential distribution
m = 18
N_particles = 2 * m**3
mu, var = log_normal_parameters(normalizing_diameter = 1)
rng = np.random.default_rng(seed=42)
sigma_distribution = rng.lognormal(mean = mu, sigma = var, size = N_particles)

# Plot histogram
custom_ticks = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
plt.figure(figsize=(8, 5))
plt.hist(sigma_distribution, bins=250, color='skyblue', edgecolor='black', alpha=0.8)
plt.xlabel("Particle Diameter (nm)", fontsize=12)
plt.xticks(custom_ticks)
plt.xlim(0, 2000)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('figures/log_normal_distribution.png')

# Sample particle sizes from exponential distribution
mu, var = log_normal_parameters()
rng = np.random.default_rng(seed=42)
sigma_distribution = rng.lognormal(mean = mu, sigma = var, size = N_particles)
while np.any(sigma_distribution > 7):
    # Resample just the outliers
    n_resample = np.sum(sigma_distribution > 7)
    sigma_distribution[sigma_distribution > 7] = rng.lognormal(mean=mu, sigma=var, size=n_resample)

# Plot histogram
custom_ticks = [0, 1, 2, 3, 4, 5, 6, 7]
plt.figure(figsize=(8, 5))
plt.hist(sigma_distribution, bins=250, color='skyblue', edgecolor='black', alpha=0.8)
plt.xlabel("Particle Diameter (nm)", fontsize=12)
plt.xticks(custom_ticks)
plt.xlim(0, 7)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('figures/normalized_log_normal_distribution.png')