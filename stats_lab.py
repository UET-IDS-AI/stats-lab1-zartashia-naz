import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
# Question 1 – Generate & Plot Histograms (and return data)
# -----------------------------------

def normal_histogram(n):
    """
    Generate n samples from Normal(0,1),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    # Generate samples
    data = np.random.normal(loc=0, scale=1, size=n)

    # Plot histogram
    plt.hist(data, bins=10)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Histogram of Normal Distribution (0,1)")
    plt.show()

    return data


def uniform_histogram(n):
    """
    Generate n samples from Uniform(0,10),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
     # Generate samples
    data = np.random.uniform(low=0, high=10, size=n)

    # Plot histogram
    plt.hist(data, bins=10)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Histogram of Uniform Distribution (0,10)")
    plt.show()

    return data


def bernoulli_histogram(n):
    """
    Generate n samples from Bernoulli(0.5),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    # Generate samples (0 or 1)
    data = np.random.binomial(n=1, p=0.5, size=n)

    # Plot histogram
    plt.hist(data, bins=10)
    plt.xlabel("Values (0 or 1)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Bernoulli Distribution (0.5)")
    plt.show()

    return data


# -----------------------------------
# Question 2 – Sample Mean & Variance
# -----------------------------------

def sample_mean(data):
    """
    Compute sample mean.
    """
    data = np.array(data)        # Convert to NumPy array (in case it's a list)
    n = len(data)               # Number of data points
    mean = np.sum(data) / n     # Mean formula
    return mean


def sample_variance(data):
    """
    Compute sample variance using n-1 denominator.
    """
    data = np.array(data)        # Convert to NumPy array
    n = len(data)               # Number of data points
    
    mean = sample_mean(data)    # First calculate mean
    
    # Apply sample variance formula
    variance = np.sum((data - mean) ** 2) / (n - 1)
    
    return variance


# -----------------------------------
# Question 3 – Order Statistics
# -----------------------------------

def order_statistics(data):
    """
    Return:
    - min
    - max
    - median
    - 25th percentile (Q1)
    - 75th percentile (Q3)

    Use a consistent quartile definition. The tests for the fixed
    dataset [5,1,3,2,4] expect Q1=2 and Q3=4.
    """
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    # Minimum and Maximum
    minimum = sorted_data[0]
    maximum = sorted_data[-1]
    
    # Median
    if n % 2 == 1:
        median = sorted_data[n // 2]
    else:
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    
    # Quartiles (Q1 and Q3)
    q1 = sorted_data[n // 4]
    q3 = sorted_data[(3 * n) // 4]
    
    return (minimum, maximum, median, q1, q3)


# -----------------------------------
# Question 4 – Sample Covariance
# -----------------------------------

def sample_covariance(x, y):
    """
    Compute sample covariance using n-1 denominator.
    """
    n = len(x)
    
    # Step 1: Compute means
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    # Step 2: Compute sum of products
    cov_sum = 0
    for i in range(n):
        cov_sum += (x[i] - mean_x) * (y[i] - mean_y)
    
    # Step 3: Divide by (n - 1)
    covariance = cov_sum / (n - 1)
    
    return covariance


# -----------------------------------
# Question 5 – Covariance Matrix
# -----------------------------------

def covariance_matrix(x, y):
    """
    Return 2x2 covariance matrix:
        [[var(x), cov(x,y)],
         [cov(x,y), var(y)]]
    """
    var_x = sample_variance(x)
    var_y = sample_variance(y)
    cov_xy = sample_covariance(x, y)
    
    return [
        [var_x, cov_xy],
        [cov_xy, var_y]
    ]

