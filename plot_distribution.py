import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

def plot_distribution(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    # Load the scores
    print(f"Loading data from {file_path}...")
    d_scores = np.load(file_path)
    
    # Create the figure
    plt.figure(figsize=(10, 6))
    
    # 1. Plot the histogram of the actual data
    # density=True makes it a probability density plot so we can overlay the bell curve
    plt.hist(d_scores, bins=50, density=True, alpha=0.6, color='royalblue', edgecolor='black', label='D_forget Scores Histogram')
    
    # 2. Fit a normal distribution (Gaussian) to the data to see how it looks
    mean, std = norm.fit(d_scores)
    
    # 3. Plot the Probability Density Function (PDF) of the fitted Gaussian
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean, std)
    plt.plot(x, p, 'k', linewidth=2.5, label=f'Gaussian Fit\n$\mu={mean:.2f}$, $\sigma={std:.2f}$')
    
    # 4. Draw a vertical red dashed line at the mean (x0)
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean ($x_0$)')
    
    # Add labels and formatting
    plt.title('Distribution of $D_{forget}$ OCSVM Scores ($d_H(x)$)')
    plt.xlabel('OCSVM Score / Distance ($d_H(x)$)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    
    # Save the plot to an image file
    output_image = file_path.replace('.npy', '_plot.png')
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved to: {output_image}")

if __name__ == "__main__":
    file_to_plot = "./weight_results_d_forget_scores.npy"
    plot_distribution(file_to_plot)
