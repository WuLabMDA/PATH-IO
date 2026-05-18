import sys, os, glob
import math
import numpy as np
from PIL import Image

import openslide
from multiprocessing import Pool
import time
import pdb
from preprocess import apply_image_filters, tissue_percent, mask_percent


def get_downsampled_image(slide, scale_factor=32):
    large_w, large_h=slide.dimensions
    new_w = math.floor(large_w/scale_factor)
    new_h = math.floor(large_h/scale_factor)    
    level = slide.get_best_level_for_downsample(scale_factor)
    whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    whole_slide_image = whole_slide_image.convert("RGB")
    img = whole_slide_image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    return img, new_w, new_h

def get_start_end_coordinates(x, tile_size):
    start = int(x * tile_size)
    end = int((x+1) * tile_size)
    return start, end

def wsi_patch(start_h, end_h, start_w, end_w, tile_size, savepath, slide_list, num_tiles_w, num_tiles_h, idx, N, prop):
    tile_region = tissue[start_h:end_h, start_w:end_w]
    tile = slide.read_region((start_w, start_h), 0, (tile_size, tile_size))
    tile = tile.convert("RGB")

    
    tile_path = savepath + slide_list + '_' + str(tile_size) + '_x'+ str(start_w) + '_'+ str(int(start_w/tile_size))+ \
    '_'+str(num_tiles_w) + '_y'+str(start_h)+'_'+str(int(start_h/tile_size))+'_'+str(num_tiles_h)+'.png'
    
    
    if prop == '40':
        tile = tile.resize((256, 256), Image.Resampling.LANCZOS)
    tile.save(tile_path)


if __name__ == '__main__':
    
    '''
    example: python svs2tile.py 0 20
    '''
    
    slide_rootpath = '/path/to/wsi_directory/'     # Input WSI directory
    out_rootpath   = '/path/to/output_directory/'  # Output directory


    
    if not os.path.exists(out_rootpath):
        os.makedirs(out_rootpath)  
    
    slide_list = [os.path.split(i)[1][:-4] for i in glob.glob(slide_rootpath+'*.svs')]
    PATCH_FUN = 'CUSTOM'

    start = sys.argv[1]
    stop = sys.argv[2]
    if int(stop) > len(slide_list):
        stop = str(len(slide_list))

    scale_factor = 1

    for i in range (int(start), int(stop)):
        savepath = out_rootpath + slide_list[i] + '/'
        
        if os.path.exists(savepath):
            print('Slide %s is already processed. Skipping.' % slide_list[i])
            continue
            
        slide_path=slide_rootpath + slide_list[i] + '.svs'
        slide = openslide.open_slide(slide_path)
        prop = slide.properties
        start_time = time.time()
        
        if not os.path.exists(savepath):
            os.makedirs(savepath) 

        if prop['aperio.AppMag']=='40':
            print('AppMag is 40')
            tile_size=512
        elif prop['aperio.AppMag']=='20':
            print('AppMag is 20')
            tile_size=256
        else:
            print('AppMag is %s for %s' % (prop['aperio.AppMag'], slide_list[i]))
            continue

        img, new_w, new_h = get_downsampled_image(slide, scale_factor=scale_factor)
        tissue=(np.array(img))

        small_tile_size = int(tile_size)
        num_tiles_h = new_h//small_tile_size
        num_tiles_w = new_w//small_tile_size

        # pdb.set_trace()

        print('Start for %s' % slide_list[i])

        PATCH_FUN = 'CUSTOM'

        if PATCH_FUN == 'CUSTOM':
            #-----------------------------------customized function to cut whole slide into png and save as hd5------------------------------
            idx = 0
            start_h_vec = np.zeros(num_tiles_h*num_tiles_w, dtype = int)
            end_h_vec = np.zeros(num_tiles_h*num_tiles_w, dtype = int)
            start_w_vec = np.zeros(num_tiles_h*num_tiles_w, dtype = int)
            end_w_vec = np.zeros(num_tiles_h * num_tiles_w, dtype = int)
            for h in range(num_tiles_h):
                for w in range(num_tiles_w):
                    start_h, end_h = get_start_end_coordinates(h, tile_size)
                    start_w, end_w = get_start_end_coordinates(w, tile_size)
                
                    start_h_vec[idx] = start_h
                    end_h_vec[idx] = end_h
                    start_w_vec[idx] = start_w
                    end_w_vec[idx] = end_w
                    idx = idx+1

            print('total patches: %s' % idx)

            with Pool() as pool:
                # prepare arguments
#                 pdb.set_trace()
                items = [((start_h_vec[kk], end_h_vec[kk], start_w_vec[kk], end_w_vec[kk], tile_size, savepath, slide_list[i], num_tiles_w, num_tiles_h, kk, idx, prop['aperio.AppMag'])) for kk in range(idx)]
                # call the same function with different data in parallel
                pool.starmap(wsi_patch, items)
            
#             mark_slide_as_processed(slide_name, out_rootpath)
            
            print('Done for %s' % slide_list[i])
            print("--- %s seconds ---" % (time.time() - start_time))

