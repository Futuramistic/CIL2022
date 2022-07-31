"""
Calculate non-maximum-suppressed pickle files from optimal brush radius files of the specified dataset for the
supervised reinforcement learning setting
"""

import argparse
import cv2
from multiprocessing import Process, cpu_count
import numpy as np
import os
import pickle
from PIL import Image
from tqdm import tqdm
import shutil


def run_cpu_tasks_in_parallel(tasks):
    """
    Run the specified tasks in parallel
    Args:
        tasks (list of Function): tasks to run
    """
    # adapted from https://stackoverflow.com/a/56138825
    running_tasks = [Process(target=task) for task in tasks]
    for running_task in running_tasks:
        running_task.start()
    for running_task in running_tasks:
        running_task.join()


def new_dir(dir_name):
    """
    Delete the specified directory recursively if it already exists, then unconditionally (re-)create the specified
    directory
    Args:
        dir_name (str): path to directory
    """
    if dir_name is not None:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.makedirs(dir_name)


def process_files(file_names, optimal_brush_radius_dir, non_max_suppressed_dir, radius):
    """
    Calculate non-maximum-suppressed pickle files from optimal brush radius files of the specified dataset for the
    supervised reinforcement learning setting
    Args:
        file_names (list of str): list of pickle files with optimal brush radii for the training samples
        optimal_brush_radius_dir (str): path to directory with optimal brush radius files
        non_max_suppressed_dir (str): path to directory into which to store the resulting non-max-suppressed files
        radius (int): radius to use for the non-maximum suppression algorithm
    """
    print(file_names)
    for file_name in file_names:
        with open(os.path.join(optimal_brush_radius_dir, os.path.splitext(file_name)[0] + '.pkl'), 'rb') as f:
            optimal_brush_radii = pickle.load(f)
        dims = (optimal_brush_radii.shape[0], optimal_brush_radii.shape[1])

        grid_x, grid_y = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), indexing='xy')

        nonzero_ys, nonzero_xs = np.nonzero(optimal_brush_radii)

        optimal_brush_radii_output = np.copy(optimal_brush_radii)
        last_y = -1
        for y, x in zip(nonzero_ys, nonzero_xs):
            if y % 10 == 0 and y != last_y:
                print(f'Sample "{file_name}": just saw y={y}')
                last_y = y

            distance_map = np.square(grid_y - y) + np.square(grid_x - x) - radius ** 2.0
            optimal_brush_radii_clone = np.copy(optimal_brush_radii)
            optimal_brush_radii_clone[distance_map >= 0] = -1
            if optimal_brush_radii_clone.max() > optimal_brush_radii[y, x]:
                optimal_brush_radii_output[y, x] = 0
            else:
                optimal_brush_radii_output[y, x] = 1

        with open(os.path.join(non_max_suppressed_dir, os.path.splitext(file_name)[0] + '.pkl'), 'wb') as f:
            pickle.dump(optimal_brush_radii_output, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        img = Image.fromarray((optimal_brush_radii_output.astype(np.float32)) * 255.0)
        img.convert('L').save(os.path.join(non_max_suppressed_dir, os.path.splitext(file_name)[0] + '.png'))
        print(f'Processed "{file_name}"')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True, type=str, help='Name of the dataset to process')
    parser.add_argument('-r', '--radius', required=False, type=int, default=10,
                        help='Radius to use for the non-maximum suppression algorithm')
    parser.add_argument('-t', '--num_tasks', required=False, type=int, default=10,
                        help='Number of tasks to run (samples to process) in parallel, '
                             'upper-bounded by the number of CPU cores on this PC')
    options = parser.parse_args()

    dataset_name = options.dataset
    radius = options.radius
    num_tasks = options.num_tasks
    dataset_path = os.path.join('dataset', dataset_name)
    
    non_max_suppressed_dir = os.path.join(dataset_path, 'training', 'non_max_suppressed')
    new_dir(non_max_suppressed_dir)

    optimal_brush_radius_dir = os.path.join(dataset_path, 'training', 'opt_brush_radius')
    if not os.path.isdir(optimal_brush_radius_dir):
        print(f'Directory not found: {optimal_brush_radius_dir}; please make sure to run opt_brush_radius_calculator.py'
              f' on the "{dataset_name}" dataset first!')
        exit()

    files_to_process = list(filter(lambda fn: fn.lower().endswith('.pkl'), os.listdir(optimal_brush_radius_dir)))
    task_files = []
    num_tasks = min(options.num_tasks, cpu_count())
    files_per_task_rounded = int(np.ceil(len(files_to_process) / num_tasks))
    num_handled_files = 0
    num_remaining_files = len(files_to_process)
    for task_idx in range(num_tasks):
        num_task_files = min(num_remaining_files, files_per_task_rounded)
        task_files.append(files_to_process[num_handled_files:num_handled_files + num_task_files])
        num_handled_files += num_task_files
        num_remaining_files = max(0, num_remaining_files - num_task_files)
    
    if num_tasks <= 1:
        # serial processing:
        for task_idx in range(num_tasks):
            process_files(task_imgs[task_idx], optimal_brush_radius_dir, non_max_suppressed_dir, radius)
    else:
        # parallel processing:
        run_cpu_tasks_in_parallel([lambda task_idx=task_idx: process_files(task_files[task_idx],
                                                                           optimal_brush_radius_dir,
                                                                           non_max_suppressed_dir, radius)
                                   for task_idx in range(num_tasks)])