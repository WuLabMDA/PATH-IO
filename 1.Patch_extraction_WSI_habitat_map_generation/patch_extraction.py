import os
import sys
import glob
import time
import math
import numpy as np
from PIL import Image
import openslide
from multiprocessing import Pool


def get_downsampled_image(slide, scale_factor=1):
    large_w, large_h = slide.dimensions
    new_w = math.floor(large_w / scale_factor)
    new_h = math.floor(large_h / scale_factor)

    level = slide.get_best_level_for_downsample(scale_factor)
    whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    whole_slide_image = whole_slide_image.convert("RGB")
    img = whole_slide_image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    return img, new_w, new_h


def get_start_coordinate(x, tile_size):
    return int(x * tile_size)


def wsi_patch(start_h, start_w, tile_size, savepath, slide_name,
              num_tiles_w, num_tiles_h, app_mag, slide_path):

    slide = openslide.open_slide(slide_path)

    tile = slide.read_region((start_w, start_h), 0, (tile_size, tile_size))
    tile = tile.convert("RGB")

    tile_path = os.path.join(
        savepath,
        f"{slide_name}_{tile_size}"
        f"_x{start_w}_{int(start_w / tile_size)}_{num_tiles_w}"
        f"_y{start_h}_{int(start_h / tile_size)}_{num_tiles_h}.png"
    )

    if app_mag == "40":
        tile = tile.resize((256, 256), Image.Resampling.LANCZOS)

    tile.save(tile_path)
    slide.close()


if __name__ == "__main__":

    """
    Example:
    python svs2tile.py 0 20
    """

    slide_rootpath = "/path/to/wsi_directory/"      # Input WSI directory
    out_rootpath = "/path/to/output_directory/"     # Output patch directory

    os.makedirs(out_rootpath, exist_ok=True)

    slide_paths = sorted(glob.glob(os.path.join(slide_rootpath, "*.svs")))
    slide_list = [os.path.splitext(os.path.basename(i))[0] for i in slide_paths]

    start = int(sys.argv[1])
    stop = int(sys.argv[2])

    if stop > len(slide_list):
        stop = len(slide_list)

    scale_factor = 1

    for i in range(start, stop):

        slide_name = slide_list[i]
        slide_path = slide_paths[i]
        savepath = os.path.join(out_rootpath, slide_name)

        if os.path.exists(savepath):
            print(f"Slide {slide_name} is already processed. Skipping.")
            continue

        os.makedirs(savepath, exist_ok=True)

        slide = openslide.open_slide(slide_path)
        prop = slide.properties
        start_time = time.time()

        app_mag = prop.get("aperio.AppMag", None)

        if app_mag == "40":
            print("AppMag is 40")
            tile_size = 512
        elif app_mag == "20":
            print("AppMag is 20")
            tile_size = 256
        else:
            print(f"AppMag is {app_mag} for {slide_name}. Skipping.")
            slide.close()
            continue

        _, new_w, new_h = get_downsampled_image(slide, scale_factor=scale_factor)

        num_tiles_h = new_h // tile_size
        num_tiles_w = new_w // tile_size

        slide.close()

        print(f"Start for {slide_name}")

        items = []

        for h in range(num_tiles_h):
            for w in range(num_tiles_w):
                start_h = get_start_coordinate(h, tile_size)
                start_w = get_start_coordinate(w, tile_size)

                items.append(
                    (
                        start_h,
                        start_w,
                        tile_size,
                        savepath,
                        slide_name,
                        num_tiles_w,
                        num_tiles_h,
                        app_mag,
                        slide_path,
                    )
                )

        print(f"Total patches: {len(items)}")

        with Pool() as pool:
            pool.starmap(wsi_patch, items)

        print(f"Done for {slide_name}")
        print(f"--- {time.time() - start_time:.2f} seconds ---")
