import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import random
import open3d as o3d
import csv
import pathlib
import shutil


def get_filepaths(folder):
    """
    Recursively get all CSV-, PLY-, and TXT file paths from specified folder.

    Parameters:
        folder (string): Desired folder to look for paths.

    Returns:
        file_paths(list): List of all found CSV-, PLY-, and TXT paths.
    """
    file_paths = glob.glob(f'{folder}/**/*_pc.csv', recursive=True)
    file_paths.extend(glob.glob(f'{folder}/**/*_pc.txt', recursive=True))
    file_paths.extend(glob.glob(f'{folder}/**/*_pc.ply', recursive=True))
    return file_paths


def get_number_of_points(file_path):
    """
    Reads CSV file (point cloud) and counts number of rows (points).
    
    Parameters:
        file_path (string): Path to a CSV (Point Cloud).

    Returns:
        points_count(int): Number of rows (Points) found in CSV (Point Cloud).
    """
    points_count = 0
    with open(file_path) as f:
        reader = csv.reader(f, delimiter=',')
        points_count = sum(1 for row in reader)
        return points_count

def subsampling_algorithm(main_folder, output_folder, point_limit, spatial_parameter):
    """
    Uses CloudCompare to subsample point clouds.

    Parameters:
        main_folder (string): Path to main folder of the dataset.
        output_folder (string): Path to folder where resampled pointclouds are stored.
        parameter_value (int/float): Parameter is based upon algorithm.
            If SPATIAL, parameter is minimum distance between points.
            If RANDOM, parameter is a fixed number that should be subsampled to.

    """

    os_command = ''
    if os.name == 'nt': # Windows
        os_command = 'CloudCompare'
    elif os.name == 'posix': # Linux
        os_command = 'cloudcompare.CloudCompare'

    non_subsampled_filepaths = get_filepaths(main_folder)
    for filepath in non_subsampled_filepaths:
        filename = os.path.basename(filepath).replace(" ", "_")
        file_parentfolder = pathlib.PurePath(filepath).parent.name.replace(" ", "_")
        output_filepath = os.path.join(output_folder, file_parentfolder)
        if not os.path.isdir(output_filepath):
            os.mkdir(output_filepath)

        output_filepath = os.path.join(output_filepath, filename)

        parameter_value = random.uniform(spatial_parameter, spatial_parameter + 0.0005)
        while(True):
            # Spatial subsampling
            os.system(f'{os_command} -SILENT -O "{filepath}" -AUTO_SAVE OFF -C_EXPORT_FMT ASC -EXT TXT -PREC 6 -SEP COMMA -SS SPATIAL {parameter_value} -SAVE_CLOUDS FILE "{output_filepath}"') 
            
            # Check if path exists, the subsampling method doesn't save a new file if it can't subsample.
            point_count = 0
            print(parameter_value)
            if os.path.exists(f'{output_filepath}'):
                point_count = get_number_of_points(f'{output_filepath}')

            # Lower the threshold and try and subsample again.
            if point_count < point_limit and parameter_value == 0.00001:
                if os.path.exists(f'{output_filepath}'):
                    os.remove(f"{output_filepath}")
                shutil.copyfile(filepath, output_filepath)
                break
            elif point_count < point_limit:
                if os.path.exists(f'{output_filepath}'):
                    os.remove(f"{output_filepath}")
                parameter_value -= 0.00009
                if parameter_value <= 0.0:
                    parameter_value = 0.00001
            else:
                break

        # Random subsampling
        os.system(f'{os_command} -SILENT -O "{output_filepath}" -AUTO_SAVE OFF -C_EXPORT_FMT ASC -EXT TXT -PREC 6 -SEP COMMA -SS RANDOM {point_limit} -SAVE_CLOUDS FILE "{output_filepath}"')

def delete_pointclouds_below_limit(main_folder, limit):
    """
    Discards point clouds that are below the limit of desired points. 

    Parameters:
        main_folder (string): Path to main folder of the dataset.
        limit (int): The limit of points desired for a point cloud.
    """

    if not input_function(rf'Delete pointclouds containing less than {limit} points? (y/n): '):
        return

    print('[INFO] Deleting Pointclouds below limit...')
    filepaths = get_filepaths(main_folder)
    for file in tqdm(filepaths):
        old_f = pd.read_csv(f'{file}')
        if len(old_f)+1 < limit:
            os.remove(file)
            print(rf'Deleted {file}')


def input_function(message):
    answer = input(message).lower()
    while(True):
        if answer == 'y':
            return True
        elif answer == 'n':
            return False
        else:
            answer = input(message).lower()


if __name__ == '__main__':
    main_folder = input('Path to dataset: ')
    cloud_compare_folder = input('Path to CloudCompare: ')

    if not os.path.isdir(cloud_compare_folder):
        os.mkdir(cloud_compare_folder)
        
    os.chdir(cloud_compare_folder)

    subsampling_algorithm(main_folder, cloud_compare_folder, 512, 0.002)

    delete_pointclouds_below_limit(cloud_compare_folder, 512)

    print('Subsampling Complete!')