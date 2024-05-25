import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Function to generate logistic distribution
def logistic_function(x, L=1, x0=0.25, k=10):
    return L / (1 + np.exp(-k * (x - x0)))

# Generate logistic distribution values
x_values = np.linspace(0, 2, 1000)
y_values = logistic_function(x_values, L=1, x0=0.25, k=10)

# Normalize to make it a probability distribution
y_values = y_values / np.sum(y_values)

# Plot the distribution
plt.plot(x_values, y_values)
plt.title('Logistic Distribution of Stamp Values')
plt.xlabel('Value ($)')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()

# Simulate stamp values based on logistic distribution
np.random.seed(42)
n_stamps = 10000
stamp_values = np.random.choice(x_values, size=n_stamps, p=y_values)

# Calculate expected value per stamp
expected_value_per_stamp = np.mean(stamp_values)

# Sample data
sample_data = np.array([16,18,20,4,26,7,23,5,0,6,2,14,53,4,12,32,15,11,0,0,0,0,7,7,0,2,2,31])

# Descriptive statistics
mean_sample = np.mean(sample_data)
std_sample = np.std(sample_data, ddof=1)  # using ddof=1 for sample standard deviation

# Parameters for confidence interval calculation
N = 192  # total number of pages
n = len(sample_data)  # sample size
margin_of_error = 5  # desired margin of error

# Calculate Z-score for current sample size and margin of error
finite_population_correction = np.sqrt((N - n) / (N - 1))
standard_error = std_sample / np.sqrt(n) * finite_population_correction
Z = margin_of_error / standard_error

# Calculate confidence level from Z-score
confidence_level = norm.cdf(Z) * 2 - 1  # two-tailed test

# Estimating total number of stamps
estimated_total_stamps = mean_sample * N

# Estimating total value of the collection
estimated_total_value = estimated_total_stamps * expected_value_per_stamp

# Output the results
print(f"Sample Mean: {mean_sample}")
print(f"Sample Standard Deviation: {std_sample}")
print(f"Sample Size: {n}")
print(f"Z-score for current sample size and margin of error: {Z}")
print(f"Confidence Level: {confidence_level * 100:.2f}%")
print(f"Mean number of stamps per sampled page: {mean_sample}")
print(f"Estimated total number of stamps: {estimated_total_stamps}")
print(f"Expected value per stamp: {expected_value_per_stamp}")
print(f"Estimated total value of the collection: {estimated_total_value}")
