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
    # adapted from https://stackoverflow.com/a/56138825
    running_tasks = [Process(target=task) for task in tasks]
    for running_task in running_tasks:
        running_task.start()
    for running_task in running_tasks:
        running_task.join()


def process_imgs(file_names, dataset_path):
    for file_name in file_names:
        image = cv2.imread(os.path.join(dataset_path, 'training', 'groundtruth', file_name))
        dims = (image.shape[0], image.shape[1])

        grid_x, grid_y = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), indexing='xy')

        optimal_brush_radii = np.zeros(dims)
        largest_brush_radius = 0
        nonzero_ys, nonzero_xs = np.nonzero(image[..., 0])

        last_y = -1
        for y, x in zip(nonzero_ys, nonzero_xs):
            if y % 10 == 0 and y != last_y:
                print(f'Sample "{file_name}": just saw y={y}')
                last_y = y
            pixel = image[y, x]
            consider_pixel = True
            for channel_val in pixel:
                if channel_val != 255:
                    consider_pixel = False
                    break
            if not consider_pixel:
                continue
            
            optimal_brush_radius = 0
            
            # try increasing radius until we get false positives
            # perform binary search for efficiency

            brush_radius_lower = 0  # no false positives with this radius
            brush_radius_upper = min(dims[0], dims[1]) / 4  # false positives with this radius

            while True:
                brush_radius = int((brush_radius_lower + brush_radius_upper) / 2)
                distance_map = np.square(grid_y - y) + np.square(grid_x - x) - brush_radius**2
                stroke_ball = distance_map <= 0  # boolean mask
                if image[stroke_ball].min() != 255:
                    brush_radius_upper = brush_radius
                    if brush_radius_lower + 1 == brush_radius_upper:
                        break
                else:
                    brush_radius_lower = brush_radius
                    if brush_radius_lower + 1 == brush_radius_upper:
                        break

            optimal_brush_radius = brush_radius_lower
            if optimal_brush_radius > largest_brush_radius:
                largest_brush_radius = optimal_brush_radius
            optimal_brush_radii[y, x] = optimal_brush_radius

        with open(os.path.join(optimal_brush_radius_dir, os.path.splitext(file_name)[0] + '.pkl'), 'wb') as f:
            pickle.dump(optimal_brush_radii, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        Image.fromarray((optimal_brush_radii.astype(np.float32) / largest_brush_radius) * 255.0).convert('L').save(os.path.join(optimal_brush_radius_dir, os.path.splitext(file_name)[0] + '.png'))
        print(f'Processed "{file_name}"')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', required=True, type=str, help='Path to the root '
                                                                             'directory of the dataset')
    options = parser.parse_args()

    dataset_dir = options.dataset_dir
    dataset_path = os.path.join('dataset', dataset_dir)
    optimal_brush_radius_dir = os.path.join(dataset_path, 'training', 'opt_brush_radius')
    os.makedirs(optimal_brush_radius_dir, exist_ok=True)

    imgs_to_process = list(filter(lambda fn: fn.lower().endswith('.png') and not os.path.exists(os.path.join(optimal_brush_radius_dir, fn.replace('.png', '.pkl'))),
                                  os.listdir(os.path.join(dataset_path, 'training', 'groundtruth'))))
    task_imgs = []
    num_tasks = 10  #  cpu_count() - 1  # probably bad idea to use all cores, as numpy also parallelizes some computations
    imgs_per_task_rounded = int(np.ceil(len(imgs_to_process) / num_tasks))
    num_handled_imgs = 0
    num_remaining_imgs = len(imgs_to_process)
    for task_idx in range(num_tasks):
        num_task_imgs = min(num_remaining_imgs, imgs_per_task_rounded)
        task_imgs.append(imgs_to_process[num_handled_imgs:num_handled_imgs + num_task_imgs])
        num_handled_imgs += num_task_imgs
        num_remaining_imgs = max(0, num_remaining_imgs - num_task_imgs)
    
    # serial processing:
    # for task_idx in range(num_tasks):
    #     process_imgs(task_imgs[task_idx], dataset_path)
    
    # parallel processing:
    run_cpu_tasks_in_parallel([lambda task_idx=task_idx: process_imgs(task_imgs[task_idx], dataset_path) for task_idx in range(num_tasks)])
