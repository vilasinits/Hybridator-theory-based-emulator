from scipy.stats import skew
import numpy as np
from lenspack.image.transforms import starlet2d

def get_moments(kappa_values, pdf_values):
    """
    Calculates the moments (mean, variance, skewness, kurtosis) of a probability distribution function.

    Parameters:
        kappa_values (numpy.ndarray): A 1D array of kappa values.
        pdf_values (numpy.ndarray): A 1D array of PDF values corresponding to `kappa_values`.

    Returns:
        tuple: Contains mean, variance, skewness, kurtosis, and normalization of the PDF.
    """
    norm = np.trapz(pdf_values, kappa_values)
    normalized_pdf_values = pdf_values / norm
    mean_kappa = np.trapz(kappa_values * normalized_pdf_values, kappa_values)
    variance = np.trapz((kappa_values - mean_kappa)**2 * normalized_pdf_values, kappa_values)
    third_moment = np.trapz((kappa_values - mean_kappa)**3 * normalized_pdf_values, kappa_values)
    fourth_moment = np.trapz((kappa_values - mean_kappa)**4 * normalized_pdf_values, kappa_values)
    S_3 = third_moment / variance**2.0
    K = fourth_moment / variance**2 - 3
    return mean_kappa, variance, S_3, K, norm

def get_image_patches(image_data, num_patches):
    """
    Divide the given image data into a specified number of patches.
    
    Parameters:
    image_data (numpy array): 2D array representing the image data.
    num_patches (int): Number of patches to divide the image into.
    
    Returns:
    list of numpy arrays: List containing the image patches.
    """
    # Get the number of rows and columns in the image data
    nrows, ncols = image_data.shape
    
    # Calculate the size of each patch
    patch_size = int(np.sqrt((nrows * ncols) / num_patches))
    
    patches = []
    # Iterate over the image data to extract patches
    for i in range(0, nrows, patch_size):
        for j in range(0, ncols, patch_size):
            # Extract a patch of the specified size
            patch = image_data[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
            # Stop if the desired number of patches is reached
            if len(patches) >= num_patches:
                return patches
    return patches

def jackknife_resampling_patches(patches):
    """
    Perform jackknife resampling on each patch to calculate statistical measures.
    
    Parameters:
    patches (list of numpy arrays): List of image patches.
    
    Returns:
    tuple of lists: Lists containing the means, variances, and skewnesses for each patch.
    """
    means = []
    variances = []
    skewnesses = []
    
    for patch in patches:
        # Flatten the patch to a 1D array
        patch_flat = patch.flatten()
        n = len(patch_flat)
        resample_means = np.zeros(n)
        
        # Perform jackknife resampling by excluding each element once
        for i in range(n):
            resample = np.delete(patch_flat, i)
            resample_means[i] = np.mean(resample)
        
        # Calculate the mean of the resampled means
        means.append(np.mean(resample_means))
        # Calculate the variance of the resampled means
        variances.append((n - 1) * np.var(resample_means, ddof=1))
        # Calculate the skewness of the resampled means
        skewnesses.append((n - 1) * skew(resample_means, bias=False))
    
    return means, variances, skewnesses

def jackknife_stats_patches(image_data, num_patches=10):
    """
    Calculate overall statistical measures (mean, variance, skewness) for an image using jackknife resampling on patches.
    
    Parameters:
    image_data (numpy array): 2D array representing the image data.
    num_patches (int, optional): Number of patches to divide the image into. Default is 10.
    
    Returns:
    tuple: Overall mean, variance, and skewness of the image patches.
    """
    # Get the image patches
    patches = get_image_patches(image_data, num_patches)
    # Perform jackknife resampling on the patches
    means, variances, skewnesses = jackknife_resampling_patches(patches)
    
    # Calculate overall statistical measures
    overall_mean = np.mean(means)
    overall_variance = np.mean(variances)
    overall_skewness = np.mean(skewnesses)
    
    return overall_mean, overall_variance, overall_skewness