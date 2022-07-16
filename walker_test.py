import argparse
import cv2
import matplotlib.pyplot as plt
from multiprocessing import Process, cpu_count
import numpy as np
import os
import pickle
import random
from PIL import Image, ImageDraw
from tqdm import tqdm
import shutil


def new_dir(dir_name):
    if dir_name is not None:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.makedirs(dir_name)


if __name__ == '__main__':
    dataset_path = os.path.join('dataset', 'original')

    imgs_to_process = list(filter(lambda fn: fn.lower().endswith('.png'), os.listdir(os.path.join(dataset_path, 'training', 'groundtruth'))))
    
    num_trajectories = 1
    trajectory_length = 1000

    trajectory_dir = os.path.join(dataset_path, 'training', 'trajectories')
    new_dir(trajectory_dir)

    for file_idx, file_name in enumerate(imgs_to_process):
        if file_idx < 2:
            continue

        image = cv2.imread(os.path.join(dataset_path, 'training', 'groundtruth', file_name))[..., 0]

        # load associated optimal brush radius map
        opt_brush_radius_path = os.path.join(dataset_path, 'training', 'opt_brush_radius', file_name.replace('.png', '.pkl'))

        if not os.path.isfile(opt_brush_radius_path):
            continue

        with open(opt_brush_radius_path, 'rb') as f:
            opt_brush_radii = pickle.load(f)

        # load associated non-max-suppressed 
        non_max_supp_path = os.path.join(dataset_path, 'training', 'non_max_suppressed', file_name.replace('.png', '.pkl'))
        with open(non_max_supp_path, 'rb') as f:
            non_max_suppressed = pickle.load(f)

        if not os.path.isfile(non_max_supp_path):
            continue
        
        dims = (image.shape[0], image.shape[1])
        grid_x, grid_y = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), indexing='xy')

        frame_store = []

        # idea: if there are multiple argmins, always prefer that with the lowest delta to the last angle

        last_angle = 0.0

        for trajectory_idx in range(num_trajectories):
            y, x = float(random.randint(0, dims[0])), float(random.randint(0, dims[1]))

            last_y_int, last_x_int = int(np.round(y)), int(np.round(x))
            
            is_on_road = image[last_y_int, last_x_int] == 255
            is_on_max = non_max_suppressed[last_y_int, last_x_int] > 0
            image[last_y_int, last_x_int] = 100
            current_is_visited = False

            visited = np.zeros_like(image)

            for step_idx in range(trajectory_length):
                current_opt_brush_radius = opt_brush_radii[last_y_int, last_x_int]

                #if not is_on_max or current_is_visited:
                # calculate closest max position

                distance_map = np.square(grid_y - y) + np.square(grid_x - x)
                distance_map[non_max_suppressed == 0] = np.inf
                distance_map[visited > 0] = np.inf
                
                # closest_y, closest_x = np.unravel_index(np.argmin(distance_map), distance_map.shape)

                if distance_map.min() == np.inf:
                    break

                closest_ys, closest_xs = np.where(distance_map == distance_map.min())

                lowest_delta_angle = np.nan
                lowest_delta_magnitude = np.nan
                for closest_y, closest_x in zip(closest_ys, closest_xs):
                    tmp_angle = np.arctan2(closest_y - y, closest_x - x)
                    if np.isnan(lowest_delta_angle) or np.abs(tmp_angle - last_angle) < np.abs(lowest_delta_angle - last_angle):
                        lowest_delta_angle = tmp_angle
                        lowest_delta_magnitude = np.sqrt( (closest_y - y)**2.0 + (closest_x - x)**2.0 )

                # else:
                #     # in circle of radius R around current position, check for pixel with highest opt_brush_radius and
                #     # move there
                #     # if multiple such points, choose the with the angle closest to the angle last taken

                #     distance_map = np.square(grid_y - y) + np.square(grid_x - x) - 5.0 ** 2.0
                #     opt_brush_radii_clone = np.copy(opt_brush_radii)

                #     # mask out
                #     opt_brush_radii_clone[distance_map > 0] = -1
                #     opt_brush_radii_clone[visited > 0] = -1

                #     best_ys, best_xs = np.where(opt_brush_radii_clone == opt_brush_radii_clone.max())

                #     lowest_delta_angle = np.nan
                #     for best_y, best_x in zip(best_ys, best_xs):
                #         tmp_angle = np.arctan2(best_y - y, best_x - x)
                #         if np.isnan(lowest_delta_angle) or np.abs(tmp_angle - last_angle) < np.abs(lowest_delta_angle - last_angle):
                #             lowest_delta_angle = tmp_angle
                    
                if is_on_road:
                    # paint on road 

                    distance_map_paint = np.square(grid_y - y) + np.square(grid_x - x) - current_opt_brush_radius ** 2.0
                    image[distance_map_paint <= 0] = 200
                    visited[distance_map_paint <= 0] = 1


                # take a step in that direction (at least one pixel in x, and one pixel in y direction?)
                
                angle = lowest_delta_angle
                magnitude = min(5, lowest_delta_magnitude)
                
                delta_y = np.sin(angle) * magnitude
                if np.abs(delta_y) < 0.1:
                    delta_y = np.sign(delta_y) * 0.1

                delta_x = np.cos(angle) * magnitude
                if np.abs(delta_x) < 0.1:
                    delta_x = np.sign(delta_x) * 0.1

                y = max(0.0, min(y + delta_y, float(distance_map.shape[0] - 1)))
                x = max(0.0, min(x + delta_x, float(distance_map.shape[1] - 1)))

                # we can simply gray out the visited pixels
                new_y_int, new_x_int = int(np.round(y)), int(np.round(x))
                
                is_on_road = image[new_y_int, new_x_int] in [255, 200]
                is_on_max = non_max_suppressed[new_y_int, new_x_int] == 255
                current_is_visited = visited[new_y_int, new_x_int] > 0
                
                visited[new_y_int, new_x_int] = 1

                for fill_y in range(min(last_y_int, new_y_int), max(last_y_int, new_y_int)):
                    for fill_x in range(min(last_x_int, new_x_int), max(last_x_int, new_x_int)):
                        visited[fill_y, fill_x] = 1

                last_y_int, last_x_int = new_y_int, new_x_int
                    
                curr_pos_mask = np.zeros_like(image)
                curr_pos_mask[new_y_int, new_x_int] = 1 # mark current position
                gif_frame = Image.fromarray((image * (1-visited) + visited * 100) * (1-curr_pos_mask) + curr_pos_mask * 150)
                draw = ImageDraw.Draw(gif_frame)
                draw.text((10, 10), str(step_idx), fill=255)
                draw.text((10, 40), f'y: {"%.1f" % y}; x: {"%.1f" % x}, r: {1 if is_on_road else 0}, m: {1 if is_on_max else 0}', fill=150)

                last_angle = angle
                frame_store.append(gif_frame)

        # duration is milliseconds between frames, loop=0 means loop infinite times
        frame_store[0].save('walker_test.gif', save_all=True, append_images=frame_store[1:], duration=100, loop=0)
        break
