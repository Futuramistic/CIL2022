import argparse
import cv2
from multiprocessing import Process, cpu_count
import numpy as np
import os
import pickle
from PIL import Image
from tqdm import tqdm
import shutil


RADIUS = 10


def run_cpu_tasks_in_parallel(tasks):
    # adapted from https://stackoverflow.com/a/56138825
    running_tasks = [Process(target=task) for task in tasks]
    for running_task in running_tasks:
        running_task.start()
    for running_task in running_tasks:
        running_task.join()


def new_dir(dir_name):
    if dir_name is not None:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.makedirs(dir_name)


def process_files(file_names, optimal_brush_radius_dir, non_max_suppressed_dir):
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

            distance_map = np.square(grid_y - y) + np.square(grid_x - x) - RADIUS ** 2.0
            optimal_brush_radii_clone = np.copy(optimal_brush_radii)
            optimal_brush_radii_clone[distance_map >= 0] = -1
            if optimal_brush_radii_clone.max() > optimal_brush_radii[y, x]:
                optimal_brush_radii_output[y, x] = 0
            else:
                optimal_brush_radii_output[y, x] = 1

        with open(os.path.join(non_max_suppressed_dir, os.path.splitext(file_name)[0] + '.pkl'), 'wb') as f:
            pickle.dump(optimal_brush_radii_output, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        Image.fromarray((optimal_brush_radii_output.astype(np.float32)) * 255.0).convert('L').save(os.path.join(non_max_suppressed_dir, os.path.splitext(file_name)[0] + '.png'))
        print(f'Processed "{file_name}"')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', required=True, type=str, help='Path to the root '
                                                                             'directory of the dataset')
    options = parser.parse_args()

    dataset_dir = options.dataset_dir
    dataset_path = os.path.join('dataset', dataset_dir)
    
    non_max_suppressed_dir = os.path.join(dataset_path, 'training', 'non_max_suppressed')
    new_dir(non_max_suppressed_dir)

    optimal_brush_radius_dir = os.path.join(dataset_path, 'training', 'opt_brush_radius')
    if not os.path.isdir(optimal_brush_radius_dir):
        print(f'Directory not found: {optimal_brush_radius_dir}')
        exit()

    files_to_process = list(filter(lambda fn: fn.lower().endswith('.pkl'), os.listdir(optimal_brush_radius_dir)))
    task_files = []
    num_tasks = 4  #  cpu_count() - 1  # probably bad idea to use all cores, as numpy also parallelizes some computations
    files_per_task_rounded = int(np.ceil(len(files_to_process) / num_tasks))
    num_handled_files = 0
    num_remaining_files = len(files_to_process)
    for task_idx in range(num_tasks):
        num_task_files = min(num_remaining_files, files_per_task_rounded)
        task_files.append(files_to_process[num_handled_files:num_handled_files + num_task_files])
        num_handled_files += num_task_files
        num_remaining_files = max(0, num_remaining_files - num_task_files)
    
    # serial processing:
    # for task_idx in range(num_tasks):
    #     process_files(task_imgs[task_idx], dataset_path, non_max_suppressed_dir)
    
    # parallel processing:
    run_cpu_tasks_in_parallel([lambda task_idx=task_idx: process_files(task_files[task_idx], optimal_brush_radius_dir, non_max_suppressed_dir) for task_idx in range(num_tasks)])
