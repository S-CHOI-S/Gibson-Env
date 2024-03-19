import os

import igibson
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from igibson.utils.assets_utils import get_scene_path

np.set_printoptions(threshold=np.inf)

def main():
    scene_id = "Rs"
    trav_map_original_size = 1000
    trav_map_size = 1000
    trav_map_erosion = 2
    floor_map = []
    floor_graph = []

    env_dir = "./data/g_dataset/gibson_v2_selected/Beach"

    with open(os.path.join(env_dir, "floors.txt"), "r") as f:
        floors = sorted(list(map(float, f.readlines())))
        print("floor_heights", floors)

    for f in range(len(floors)):
        trav_map = Image.open(os.path.join(env_dir, "floor_trav_{}.png".format(f)))
        obstacle_map = Image.open(os.path.join(env_dir, "floor_{}.png".format(f)))
        render_map = Image.open(os.path.join(env_dir, "floor_render_{}.png".format(f)))
        trav_map = np.array(trav_map.resize((trav_map_size, trav_map_size)))
        obstacle_map = np.array(obstacle_map.resize((trav_map_size, trav_map_size)))
        render_map = np.array(render_map.resize((trav_map_size, trav_map_size)))
        trav_map[obstacle_map == 0] = 0
        trav_map = cv2.erode(trav_map, np.ones((trav_map_erosion, trav_map_erosion)))
        plt.figure(f, figsize=(12, 12))
        # plt.imshow(render_map)

    # plt.show()
    # print(trav_map)
    # print(trav_map.shape)

    trav_map_modified = np.where(trav_map > 0, 0, 255).astype(np.uint8)
    
    alpha_channel = np.where(trav_map > 0, 0, 255).astype(np.uint8)
    overlay_rgba = Image.fromarray(np.zeros((*trav_map_modified.shape, 4), dtype=np.uint8), mode='RGBA')
    overlay_rgba.putalpha(Image.fromarray(alpha_channel))

    render_map = Image.fromarray(render_map)
    render_map.paste(overlay_rgba, (0, 0), overlay_rgba)

    render_map.save(os.path.join(env_dir, "render_obstacle_map.png"))

    plt.imshow(render_map)
    plt.show()



if __name__ == "__main__":
    main()