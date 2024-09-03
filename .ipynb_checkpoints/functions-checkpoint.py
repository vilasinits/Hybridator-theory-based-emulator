import numpy as np
# import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# import lenspack
from lenspack.image.transforms import starlet2d
from jax.numpy.fft import rfft2, irfft2, rfftfreq, fftfreq
import scipy
import jax
import jax.numpy as jnp
from functools import partial
from scipy.interpolate import CubicSpline
import astropy.units as u

def fourier_coordinate(x, y, map_size):
    return (map_size // 2 + 1) * x + y

@partial(jax.jit, static_argnums=(4,))
def calculate_Cls(map, angle, ell_min, ell_max, n_bins):
    """
    map: the image from which the angular power spectra (Cls) has to be calculated
    angle: side angle in the units of degree
    ell_min: the minimum multipole moment to get the Cls
    ell_max: the maximum multipole moment to get the Cls
    n_bins: number of bins in the ells
    """
    ell_min = jnp.array(ell_min)
    ell_max = jnp.array(ell_max)
    # n_bins = jnp.array(n_bins, int)

    # Calculate the Fourier Transforms
    map_ft = rfft2(map)
    map_ft = map_ft.flatten()
    ell_edges = jnp.linspace(ell_min, ell_max, num=n_bins+1)

    # Define pixel physical size in Fourier space
    lpix = 360.0 / angle
    # Initialize arrays to store power and hits for each ell bin
    power_l = jnp.zeros(n_bins)
    hits = jnp.zeros(n_bins)
    
    def loop_body(j, val):
        i, power_l, hits = val
        lx = jnp.minimum(i, map.shape[1] - i) * lpix
        ly = j * lpix
        l = jnp.sqrt(lx**2. + ly**2.)
        pixid = fourier_coordinate(i, j, map.shape[1])
        bin_idx = jnp.digitize(l, ell_edges) - 1
        power_l = power_l.at[bin_idx].add(jnp.abs(map_ft[pixid]**2.))
        hits = hits.at[bin_idx].add(1)
        return i, power_l, hits

    def outer_loop_body(i, val):
        _, power_l, hits = val
        _, power_l, hits = jax.lax.fori_loop(0, map.shape[0], loop_body, (i, power_l, hits))
        return i, power_l, hits

    _, power_l, hits = jax.lax.fori_loop(0, map.shape[1], outer_loop_body, (0, power_l, hits))

    # Calculate Cls based on the accumulated power and hits
    normalization = (jnp.deg2rad(angle) / (map.shape[0] * map.shape[0]))**2
    cls_values = power_l / hits
    ell_bins = 0.5 * (ell_edges[1:] + ell_edges[:-1])

    return jnp.array(ell_edges), jnp.array(ell_bins), jnp.array(cls_values * normalization)


def calculate_pdf(map, n_bins):
    counts, edges = jnp.histogram(map, bins = n_bins, density=True)
    bin_centers = 0.5*(edges[:-1]+edges[1:])
    return edges, bin_centers, counts


def calculate_cdf(pdf, edges):
    return  np.cumsum(pdf) * np.diff(edges)


def starlet_decompose(map, nscales):
    return starlet2d(map, nscales)


def generate_initial_gaussian_field(shape, sigma=2.):
    """
    Generates a Gaussian random field with a specified standard deviation.

    Parameters:
    - shape (tuple of ints): The shape of the output field, e.g., (x, y) for a 2D array.
    - sigma (float, optional): The standard deviation of the Gaussian distribution. Defaults to 2.

    Returns:
    - numpy.ndarray: A Gaussian random field as a numpy array.
    """
    return np.random.normal(0, sigma, shape)


def get_bin_edges(bin_centers):
    bin_width = jnp.diff(bin_centers).mean()
    bin_edges = jnp.concatenate([
        jnp.array([bin_centers[0] - bin_width / 2]),
        (bin_centers[:-1] + bin_centers[1:]) / 2,
        jnp.array([bin_centers[-1] + bin_width / 2])
    ])
    return bin_edges

# @partial(jax.jit, static_argnums=(3,))
def inverse_cdf_transform(image, cdf_target, bin_target, nbins):
    """
    Applies the inverse cumulative distribution function (CDF) transformation to an image.
    This is used to adjust the histogram of the input image to match a target histogram specified by the target CDF.

    Parameters:
    - image (jax.numpy.ndarray): The input image as a 2D JAX array.
    - cdf_target (jax.numpy.ndarray): The target cumulative distribution function.
    - bin_target (jax.numpy.ndarray): The bin edges associated with the target CDF.
    - nbins (int): The number of bins to use for the histogram of the input image.

    Returns:
    - jax.numpy.ndarray: The image transformed to have a histogram matching the target CDF.
    """
    hist, bin_edges = jnp.histogram(image, bins=nbins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Interpolate histogram to get smoother values for the source image
    source_bins = jnp.linspace(bin_centers[0], bin_centers[-1]+(bin_centers[1]-bin_centers[0]), 2000)
    source_spline = CubicSpline(bin_centers, hist)
    source_hist = source_spline(source_bins)
    
    # Calculate the CDF of the source histogram
    cdf_source = jnp.cumsum(source_hist) * (source_bins[1] - source_bins[0])
    
    # Interpolate target CDF and bin edges for smooth target histogram
    target_spline = CubicSpline(bin_target, cdf_target)
    target_bins = jnp.linspace(bin_target[0], bin_target[-1], 2000)
    smoothed_cdf_target = target_spline(target_bins)
    
    # Map the original image values to the new histogram
    mapped_cdf = jnp.interp(image.flatten(), source_bins, cdf_source)
    mapped_image = jnp.interp(mapped_cdf, smoothed_cdf_target, target_bins)
    
    return mapped_image.reshape(image.shape)

# @partial(jax.jit, static_argnums=(4, 5, 6))
def generate_field_with_target_cls(input_field, angle, target_cls, target_ells, ell_max=40000, n_bins=50):
    """
    Adjusts the power spectrum of an input field to match a target power spectrum.

    Parameters:
    - input_field (jax.numpy.ndarray): The input field as a 2D JAX array.
    - angle (float): The field of view in degrees.
    - target_cls (jax.numpy.ndarray): Target power spectrum values.
    - target_ells (jax.numpy.ndarray): Target multipole moments associated with target_cls.
    - ell_max (int): Maximum multipole moment to consider.
    - n_bins (int): Number of bins to use for the power spectrum calculation.

    Returns:
    - jax.numpy.ndarray: The adjusted input field, transformed to match the target power spectrum.
    """
    shape = input_field.shape
    assert len(target_cls) == len(target_ells), "target_cls and target_ells must have the same length"
    assert jnp.all(target_cls >= 0), "All target_cls values must be non-negative"
    
    # Fourier transform of the input field
    field_ft = rfft2(input_field)
    
    # Calculate the Cls for the input field
    ell_edges, ell_bins, field_cls = calculate_Cls(input_field, angle, 0, ell_max, n_bins)
    
    # Compute lpix and l values for FFT pixels
    lpix = 360.0 / angle
    lx = rfftfreq(shape[0]) * shape[0] * lpix
    ly = fftfreq(shape[1]) * shape[1] * lpix
    l = jnp.sqrt(lx[np.newaxis, :]**2 + ly[:, np.newaxis]**2)
    # new_l = jnp.linspace(l[0],l[-1],1000)
    # Interpolate Cls for the input field and the target
    field_cls_interp = interp1d(ell_bins, field_cls, kind="linear", bounds_error=False, fill_value=1e-9)
    Cl_field = field_cls_interp(l)
    target_cls_interp = interp1d(target_ells, target_cls, kind="linear", bounds_error=False, fill_value=1e-9)
    Cl_target = target_cls_interp(l)
    
    # Adjust the amplitude based on the target Cls
    adjustment_factor = jnp.sqrt(Cl_target / Cl_field)
    adjusted_amplitude = jnp.abs(field_ft) * adjustment_factor
    
    # Recombine adjusted amplitude with original phase
    adjusted_field_ft = adjusted_amplitude * jnp.exp(1j * jnp.angle(field_ft))
    
    # Inverse Fourier Transform to get the adjusted field in real space
    adjusted_field = irfft2(adjusted_field_ft)
    
    return adjusted_field

def compute_rrmse(true_values,true_bins, estimated_values, estimated_bins):
    true_spline = CubicSpline(true_bins, true_values)
    estimated_spline = CubicSpline(estimated_bins, estimated_values)
    # Determine overlapping bin range
    overlap_start = max(true_bins[0], estimated_bins[0])
    overlap_end = min(true_bins[-1], estimated_bins[-1])
    if overlap_start < overlap_end:
        overlap_bins = jnp.linspace(overlap_start, overlap_end, num=100)
        true_overlap_values = true_spline(overlap_bins)
        estimated_overlap_values = estimated_spline(overlap_bins)
        rmse = jnp.sqrt(jnp.mean((true_overlap_values - estimated_overlap_values) ** 2))
        rrmse = rmse / jnp.sqrt(jnp.sum(estimated_overlap_values ** 2))
        return rrmse


def get_target_values(data: np.ndarray, angle: u.Quantity, ell_min: int = 0, ell_max: int = 24000, cls_bins: int = 50, nscales: int = 1, pdf_bins: int = 200):
    angle_deg = angle.to(u.deg).value
    target = data - np.mean(data)
    # print("the mean of target is: ", np.mean(target))
    target_ell_edges, target_ell_bins, target_cls_values = calculate_Cls(target, angle_deg, ell_min, ell_max, cls_bins)
    target_starlet = starlet_decompose(target, nscales)
    target_edges, target_bin_centers, target_pdf = calculate_pdf(target, pdf_bins)
    target_cdf = calculate_cdf(target_pdf, target_edges) 
    
    starlet_results = process_starlet_scales(target_starlet, angle_deg, ell_min, ell_max, cls_bins, pdf_bins, nscales)

    return {
        "nscales": nscales,
        "ell_min": ell_min,
        "ell_max": ell_max,
        "cls_bins": cls_bins,
        "pdf_bins": pdf_bins,
        "angle": angle_deg,
        "target": target,
        "target_ell_edges": target_ell_edges,
        "target_ell_bins": target_ell_bins,
        "target_cls_values": target_cls_values,
        "target_starlet": target_starlet,
        "target_edges": target_edges,
        "target_bin_centers": target_bin_centers,
        "target_pdf": target_pdf,
        "target_cdf": target_cdf,
        **starlet_results
    }
    

def process_starlet_scales(target_starlet: np.ndarray, angle: float, ell_min: int, ell_max: int, cls_bins: int, pdf_bins: int, nscales: int):
    starlet_edges = []
    starlet_bin_centers = []
    starlet_pdf = []
    starlet_cdf = []
    starlet_cls = []

    for i in range(nscales + 1):
        edges, bin_centers, counts = calculate_pdf(target_starlet[i], pdf_bins)
        starlet_edges.append(edges)
        starlet_bin_centers.append(bin_centers)
        starlet_pdf.append(counts)
        starlet_cdf.append(calculate_cdf(counts, edges))
        starlet_cls.append(calculate_Cls(target_starlet[i], angle, ell_min, ell_max, cls_bins)[2])
    
    return {
        "target_starlet_edges": starlet_edges,
        "target_starlet_bin_centers": starlet_bin_centers,
        "target_starlet_pdf": starlet_pdf,
        "target_starlet_cdf": starlet_cdf,
        "target_starlet_cls": starlet_cls
    }
    
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