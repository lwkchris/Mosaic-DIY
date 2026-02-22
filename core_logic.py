import os
import re
import cv2
import numpy as np
import ctypes
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
from scipy.spatial.distance import euclidean

def safe_imread(file_path):
    try:
        img_array = np.fromfile(file_path, dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        return None

def resize(im, tile_row, tile_col):
    shape_row, shape_col = im.shape[0], im.shape[1]
    target_aspect = tile_col / tile_row
    current_aspect = shape_col / shape_row

    if current_aspect > target_aspect:
        new_width = int(target_aspect * shape_row)
        offset = (shape_col - new_width) // 2
        im_cropped = im[:, offset:offset + new_width]
    else:
        new_height = int(shape_col / target_aspect)
        offset = (shape_row - new_height) // 2
        im_cropped = im[offset:offset + new_height, :]

    return cv2.resize(im_cropped, (tile_col, tile_row), interpolation=cv2.INTER_CUBIC)

def img_distance(im1, im2):
    return euclidean(im1.flatten(), im2.flatten())

def load_all_images(img_dir, tile_row, tile_col, shared_counter):
    filenames = [f for f in os.listdir(img_dir) if re.search(r"\.(jpg|jpeg|png)$", f, re.I)]
    result = []
    for filename in filenames:
        filepath = os.path.join(img_dir, filename)
        im = safe_imread(filepath)
        if im is not None:
            result.append(np.array(resize(im, tile_row, tile_col)))
        with shared_counter.get_lock():
            shared_counter.value += 1
    return np.array(result, dtype=np.uint8)

def find_closest_image(q, shared_tile_images, tile_images_shape, shared_result, img_shape, tile_row, tile_col, shared_counter):
    tile_images = np.frombuffer(shared_tile_images, dtype=np.uint8).reshape(tile_images_shape)
    while True:
        task = q.get()
        if task is None: break
        row, col, im_roi = task
        min_dist = float("inf")
        min_img = None
        for im in tile_images:
            dist = img_distance(im_roi, im)
            if dist < min_dist:
                min_dist, min_img = dist, im
        im_res = np.frombuffer(shared_result, dtype=np.uint8).reshape(img_shape)
        if min_img is not None:
            im_res[row:row + tile_row, col:col + tile_col, :] = min_img
        with shared_counter.get_lock():
            shared_counter.value += 1
        q.task_done()

def generate_mosaic_core(infile, img_dir, ratio, num_processes, shared_counter):
    img = safe_imread(infile)
    if img is None: raise FileNotFoundError(f"Target image not found: {infile}")

    tile_row, tile_col = (120, 90) if img.shape[0] >= img.shape[1] else (90, 120)
    filenames = [f for f in os.listdir(img_dir) if re.search(r"\.(jpg|jpeg|png)$", f, re.I)]

    img_count = len(filenames)
    img_shape = [int(img.shape[0] / tile_row) * tile_row * ratio, int(img.shape[1] / tile_col) * tile_col * ratio, 3]
    total_tiles = (img_shape[0] // tile_row) * (img_shape[1] // tile_col)

    yield img_count + total_tiles

    tile_images = load_all_images(img_dir, tile_row, tile_col, shared_counter)
    if len(tile_images) == 0: raise ValueError("No valid images found.")

    img_resized = cv2.resize(img, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_CUBIC)
    shared_tile_images = RawArray(ctypes.c_ubyte, len(tile_images.flatten()))
    np.copyto(np.frombuffer(shared_tile_images, dtype=np.uint8).reshape(tile_images.shape), tile_images)
    shared_result = RawArray(ctypes.c_ubyte, img_shape[0] * img_shape[1] * 3)

    q = mp.JoinableQueue()
    processes = [mp.Process(target=find_closest_image, args=(
        q, shared_tile_images, tile_images.shape, shared_result, img_shape, tile_row, tile_col, shared_counter),
                            daemon=True) for _ in range(num_processes)]

    for p in processes: p.start()
    for row in range(0, img_shape[0], tile_row):
        for col in range(0, img_shape[1], tile_col):
            q.put([row, col, img_resized[row:row + tile_row, col:col + tile_col, :]])

    q.join()
    for _ in range(num_processes): q.put(None)
    for p in processes: p.join()

    yield np.frombuffer(shared_result, dtype=np.uint8).reshape(img_shape).copy()