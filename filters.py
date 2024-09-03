import numpy as np
from lenspack.image.transforms import starlet2d
from jax.numpy.fft import fft2, ifft2, fftshift, ifftshift
import jax
import jax.numpy as jnp

from jax import lax


def spherical_top_hat_filter(scale_exp: int, size: int):
    """
    Creates a 2D spherical top-hat filter with a given dyadic scale and size.
    
    Parameters
    ----------
    scale_exp : int
        Exponent of the scale factor (2 ** scale_exp).
    size : int
        Size of the filter kernel (must be an odd number to ensure a center pixel).
    
    Returns
    -------
    top_hat : jax.numpy.ndarray
        Normalized 2D spherical top-hat filter.
    """
    scale = 2 ** scale_exp
    filter_radius = scale
    center = size // 2

    x = jnp.arange(size) - center
    y = jnp.arange(size) - center
    X, Y = jnp.meshgrid(x, y)
    R = jnp.sqrt(X**2 + Y**2)
    top_hat = jnp.where(R <= filter_radius, 1, 0)
    
    # Normalize the kernel so that it integrates to 1
    normalization_factor = jnp.sum(top_hat)
    top_hat = top_hat / normalization_factor
    
    return top_hat

def apply_filter(image: jnp.ndarray, filter_2d: jnp.ndarray):
    """
    Applies a 2D filter to an image using convolution with proper FFT shift.
    """
    # Perform FFT on both the image and the filter, and shift the zero-frequency component to the center
    fft_image = fftshift(fft2(image))
    fft_filter = fftshift(fft2(filter_2d, s=image.shape))
    
    # Multiply the FFTs and shift back
    fft_result = fft_image * fft_filter
    ifft_result = fftshift(ifft2(ifftshift(fft_result)))
    
    return ifft_result.real

def tophatdecompose(image: jnp.ndarray, nscales: int):
    """
    Decompose an image using the top-hat filter into multiple scales.
    
    Parameters
    ----------
    image : jax.numpy.ndarray
        Input image to be decomposed.
    nscales : int
        Number of scales for decomposition.
    
    Returns
    -------
    wavelet_coeffs : jax.numpy.ndarray
        Wavelet coefficients for each scale.
    """
    max_scale_exp = nscales
    c0 = image
    c = c0
    wavelet_coeffs = jnp.zeros((nscales + 1,) + image.shape)

    def decompose_step(i, val):
        wavelet_coeffs, c = val
        kernel = spherical_top_hat_filter(i, image.shape[0])
        T_c0 = apply_filter(image, kernel)
        w = c - T_c0
        wavelet_coeffs = wavelet_coeffs.at[i-1].set(w)
        c = T_c0
        return wavelet_coeffs, c

    wavelet_coeffs, c = lax.fori_loop(1, max_scale_exp + 1, decompose_step, (wavelet_coeffs, c))
    
    wavelet_coeffs = wavelet_coeffs.at[max_scale_exp].set(c)
    return wavelet_coeffs

def starlet_decompose(map, nscales):
    return starlet2d(map, nscales)


# def spherical_top_hat_filter(scale_exp: int, size: int):
#     """
#     Creates a 2D spherical top-hat filter with a given dyadic scale and size.
    
#     Parameters
#     ----------
#     scale_exp : int
#         Exponent of the scale factor (2 ** scale_exp).
#     size : int
#         Size of the filter kernel (must be an odd number to ensure a center pixel).
    
#     Returns
#     -------
#     top_hat : jax.numpy.ndarray
#         Normalized 2D spherical top-hat filter.
#     """
#     scale = 2 ** scale_exp
#     filter_radius = scale
#     center = size // 2
#     top_hat = jnp.zeros((size, size))
    
#     def set_values(i, top_hat):
#         def set_inner_values(j, top_hat):
#             r = jnp.sqrt((i - center) ** 2 + (j - center) ** 2)
#             top_hat = lax.cond(r <= filter_radius, lambda x: top_hat.at[i, j].set(1), lambda x: top_hat, None)
#             return top_hat
        
#         top_hat = lax.fori_loop(0, size, set_inner_values, top_hat)
#         return top_hat

#     top_hat = lax.fori_loop(0, size, set_values, top_hat)
    
#     # Normalize the kernel so that it integrates to 1
#     normalization_factor = jnp.sum(top_hat)
#     top_hat /= normalization_factor
    
#     return top_hat

# def apply_filter(image: jnp.ndarray, filter_2d: jnp.ndarray):
#     """
#     Applies a 2D filter to an image using convolution with proper FFT shift.
#     """
#     # Perform FFT on both the image and the filter, and shift the zero-frequency component to the center
#     fft_image = fftshift(fft2(image))
#     fft_filter = fftshift(fft2(filter_2d, s=image.shape))
    
#     # Multiply the FFTs and shift back
#     fft_result = fft_image * fft_filter
#     ifft_result = fftshift(ifft2(ifftshift(fft_result)))
    
#     return ifft_result.real

# def tophatdecompose(image: jnp.ndarray, nscales: int):
#     max_scale_exp = nscales
#     c0 = image
#     c = c0
#     wavelet_coeffs = []

#     def decompose_step(i, val):
#         wavelet_coeffs, c = val
#         kernel = spherical_top_hat_filter(i, image.shape[0])
#         T_c0 = apply_filter(image, kernel)
#         w = c - T_c0
#         wavelet_coeffs = wavelet_coeffs.at[i-1].set(w)
#         c = T_c0
#         return wavelet_coeffs, c

#     wavelet_coeffs = jnp.zeros((nscales + 1,) + image.shape)
#     wavelet_coeffs, c = lax.fori_loop(1, max_scale_exp + 1, decompose_step, (wavelet_coeffs, c))
    
#     wavelet_coeffs = wavelet_coeffs.at[max_scale_exp].set(c)
#     return wavelet_coeffs



# def spherical_top_hat_filter(scale_exp: int, size: int):
#     """
#     Creates a 2D spherical top-hat filter with a given dyadic scale and size.
    
#     Parameters
#     ----------
#     scale_exp : int
#         Exponent of the scale factor (2 ** scale_exp).
#     size : int
#         Size of the filter kernel (must be an odd number to ensure a center pixel).
    
#     Returns
#     -------
#     top_hat : 2D jax.numpy array
#         Normalized 2D spherical top-hat filter.
#     """
#     scale = 2 ** scale_exp
#     filter_radius = scale
#     center = size // 2
#     top_hat = jnp.zeros((size, size))
    
#     for i in range(size):
#         for j in range(size):
#             r = jnp.sqrt((i - center) ** 2 + (j - center) ** 2)
#             if r <= filter_radius:
#                 top_hat = top_hat.at[i, j].set(1)
    
#     # Normalize the kernel so that it integrates to 1
#     normalization_factor = jnp.sum(top_hat)
#     top_hat /= normalization_factor
    
#     return top_hat

# def apply_filter(image: np.ndarray, filter_2d: np.ndarray):
#     """
#     Applies a 2D filter to an image using convolution with proper FFT shift.
#     """
#     # Perform FFT on both the image and the filter, and shift the zero-frequency component to the center
#     fft_image = fftshift(fft2(image))
#     fft_filter = fftshift(fft2(filter_2d, s=image.shape))
    
#     # Multiply the FFTs and shift back
#     fft_result = fft_image * fft_filter
#     ifft_result = fftshift(ifft2(ifftshift(fft_result)))
    
#     return ifft_result.real


# def tophatdecompose(image: np.ndarray, nscales: int):
#     max_scale_exp = nscales
#     c0 = image
#     c = c0
#     wavelet_coeffs = []
#     for i in range(1, max_scale_exp + 1):
#         kernel = spherical_top_hat_filter(i, image.shape[0])
#         T_c0 = apply_filter(image, kernel)
#         w = c - T_c0
#         wavelet_coeffs.append(w)
#         c = T_c0
#     wavelet_coeffs.append(c)
#     return wavelet_coeffs