import numpy as np
import matplotlib.pyplot as plt
from functions import *
from load_data import *


def hybridate(redshift, file_number, model_name="model_one",ell_min=0, ell_max=24000, cls_bins=50, nscales=1, pdf_bins=200, no_iterations=50, plot=False):
    data, angle_ = load_data_mnu(redshift, file_number, model_name)
    # data, angle_ = load_data(file_no=75)
    target_results = get_target_values(data, angle_, ell_min, ell_max, cls_bins, nscales, pdf_bins)

    target = target_results["target"]
    angle = target_results["angle"]
    nscales = target_results["nscales"]
    ell_min = target_results["ell_min"]
    ell_max = target_results["ell_max"]
    cls_bins = target_results["cls_bins"]
    pdf_bins = target_results["pdf_bins"]
    target_starlet_cls = target_results["target_starlet_cls"]
    target_ell_edges = target_results["target_ell_edges"]
    target_ell_bins = target_results["target_ell_bins"]
    target_cls_values = target_results["target_cls_values"]
    target_starlet = target_results["target_starlet"]
    target_edges = target_results["target_edges"]
    target_bin_centers = target_results["target_bin_centers"]
    target_pdf = target_results["target_pdf"]
    target_cdf = target_results["target_cdf"]
    target_starlet_edges = target_results["target_starlet_edges"]
    target_starlet_bin_centers = target_results["target_starlet_bin_centers"]
    target_starlet_pdf = target_results["target_starlet_pdf"]
    target_starlet_cdf = target_results["target_starlet_cdf"]

    gaussian_map = generate_initial_gaussian_field(target.shape, 0.015)
    initial_map = gaussian_map
    
    if plot:
       fig, axs = plt.subplots(1, nscales + 3, figsize=(22, 5)) 
       
    for i in range(no_iterations):
        corrected_cls_map = generate_field_with_target_cls(initial_map, angle, target_cls_values, target_ell_bins, ell_max, cls_bins)
        initial_starlet = starlet2d(initial_map, nscales)
        corrected_pdf_starlet_map = []
        for scale in range(len(target_starlet)):
            corrected_pdf_starlet_map.append(inverse_cdf_transform(initial_starlet[scale], target_starlet_cdf[scale], target_starlet_bin_centers[scale], pdf_bins))

        corrected_pdf_map = np.sum(corrected_pdf_starlet_map, axis=0)
        corrected_full_map = 0.5 * (corrected_cls_map + corrected_pdf_map)
        corrected_full_map = corrected_full_map - np.mean(corrected_full_map)
        initial_map = corrected_full_map
        
        if plot:
            corrected_ell_edges, corrected_ell_bins, corrected_cls_values = calculate_Cls(corrected_full_map, angle, ell_min, ell_max, cls_bins)
            corrected_edges, corrected_bin_centers, corrected_pdf = calculate_pdf(corrected_full_map, target_edges)

            corrected_starlet = starlet2d(corrected_full_map, nscales)
            corrected_starlet_edges = []
            corrected_starlet_bin_centers = []
            corrected_starlet_pdf = []
            corrected_starlet_cls = []

            for scale in range(nscales + 1):
                edges, bin_centers, counts = calculate_pdf(corrected_starlet[scale], target_starlet_edges[scale])
                corrected_starlet_edges.append(edges)
                corrected_starlet_bin_centers.append(bin_centers)
                corrected_starlet_pdf.append(counts)
                corrected_starlet_cls.append(calculate_Cls(corrected_starlet[scale], angle, ell_min, ell_max, cls_bins)[2])
    
            axs[0].scatter(i, compute_rrmse(target_cls_values, target_ell_bins, corrected_cls_values, corrected_ell_bins))
            axs[1].scatter(i, compute_rrmse(target_pdf, target_bin_centers, corrected_pdf, corrected_bin_centers))
            for scale in range(nscales + 1):
                axs[2+scale].scatter(i, compute_rrmse(target_starlet_pdf[scale], target_starlet_bin_centers[scale], corrected_starlet_pdf[scale], corrected_starlet_bin_centers[scale]))
                axs[2+scale].set_xlabel("Iterations", fontsize=18)
                axs[2+scale].set_ylabel("RRMSE", fontsize=18)
                axs[2+scale].set_yscale("log")  
                
            axs[0].set_xlabel("Iterations", fontsize=18)
            axs[0].set_ylabel("RRMSE", fontsize=18)
            axs[0].set_yscale("log")
            axs[1].set_xlabel("Iterations", fontsize=18)
            axs[1].set_ylabel("RRMSE", fontsize=18)
            axs[1].set_yscale("log")  
            
    return { 
            "hybridated_map": initial_map,
            **target_results
    }