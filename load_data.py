import os
from astropy.io import fits
import astropy.units as u
from skimage.transform import downscale_local_mean

import sys
import numpy as np

def load_data_mnu(redshift, file_number, model_name):
    # Define the base folder for all models
    parent_folder = "/feynman/work/dap/lcs/vt272285/data/sim_data/MassiveNus/"

    # Model parameters for each model
    models = {
        "model_one": {"mnv": 0.00000, "om": 0.30000, "As": 2.1000},
        "model_two": {"mnv": 0.09041, "om": 0.28447, "As": 2.1757},
        "model_three": {"mnv": 0.10000, "om": 0.30000, "As": 2.1000},
        "model_four": {"mnv": 0.11874, "om": 0.31434, "As": 2.0079},
    }

    # Dynamically generate paths for each model
    def generate_paths(models):
        redshifts = [0.5, 1.0, 1.5, 2.0, 2.5]
        for model_name, params in models.items():
            base_name = f"convergence_gal_mnv{params['mnv']:.5f}_om{params['om']:.5f}_As{params['As']:.4f}"
            paths = {z: os.path.join(parent_folder, f"{base_name}/Maps{int(z*10)}") for z in redshifts}
            models[model_name]["paths"] = paths

    # Populate model dictionaries with paths
    generate_paths(models)

    def get_file_name(model_name, redshift, file_number, models):
        try:
            model = models[model_name]
            folder_path = model["paths"][redshift]
            filename = f"WLconv_z{redshift:.2f}_{file_number:04d}r.fits"
            return os.path.join(folder_path, filename)
        except KeyError as e:
            raise ValueError(f"Missing data: {e}")
        
    try:
        # model_name = "model_one"
        # redshift = 1.5  #[0.5, 1.0, 1.5, 2.0, 2.5]
        # file_number = 100 #from 0 til 9999
        full_file_path = get_file_name(model_name, redshift, file_number, models)
        # print(full_file_path)
    except ValueError as err:
        print(err)
    with fits.open(full_file_path) as hdu:
        data = hdu[0].data
        angle_ = hdu[0].header["ANGLE"] * u.deg
        
    return data, angle_

def load_data(file_no):
    base_path = "/feynman/work/dap/lcs/vt272285/HOWLS/HOWLS_DATA/"
    fiducial = "SLICS_LCDM/"
    fiducial_path = base_path + fiducial + "kappa_noise_GalCatalog_LOS_cone" + str(file_no) + ".fits_s333" + str(file_no) + "_zmin0.0_zmax3.0_sys_3.fits_ks_nomask_shear.fits"
    full_file_path = fiducial_path
    try:
        with fits.open(full_file_path) as hdu:
            # print("file found! and path is:", full_file_path)
            data = hdu[0].data
            data = data[0]
            angle_ = 10 * u.deg
        return data, angle_
    except FileNotFoundError:
        print(f"File {full_file_path} not found.")
        return None
    
def load_data_slics(file_no):
    args = sys.argv
    fname_in = args[1]
    npix = 7745

    fname1 = '/feynman/work/dap/lcs/vt272285/data/SLICS/2.007kappa_weight.dat_LOS400'
    # Read binary file into den_map
    with open(fname1, 'rb') as f1:
        data_bin = np.fromfile(f1, dtype=np.float32)
        den_map = np.reshape(np.float32(data_bin), [npix, npix]) 
        
    # Fix normalization
    den_map *= 64.0
    den_map = den_map[:-1,:-1]
    kappa_downscaled = downscale_local_mean(den_map-np.mean(den_map), (16, 16))
    angle_ = 10*u.deg
    return kappa_downscaled, angle_
