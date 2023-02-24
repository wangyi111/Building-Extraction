import numpy as np
from PIL import Image
from tqdm import tqdm
import os

def crop_tiles(h, w, hTile, wTile):
    # Number of tiles
    nTilesX = np.uint8(np.ceil(w / wTile))
    nTilesY = np.uint8(np.ceil(h / hTile))

    # Total remainders
    remainderX = nTilesX * wTile - w
    remainderY = nTilesY * hTile - h

    # Set up remainders per tile
    remaindersX = np.ones((nTilesX-1, 1)) * np.uint16(np.floor(remainderX / (nTilesX-1)))
    remaindersY = np.ones((nTilesY-1, 1)) * np.uint16(np.floor(remainderY / (nTilesY-1)))
    remaindersX[0:np.remainder(remainderX, np.uint16(nTilesX-1))] += 1
    remaindersY[0:np.remainder(remainderY, np.uint16(nTilesY-1))] += 1        

    # Initialize array of tile boxes
    tiles = np.zeros((nTilesX * nTilesY, 4), np.uint16)

    # Determine proper tile boxes
    k = 0
    x = 0
    for i in range(nTilesX):
        y = 0
        for j in range(nTilesY):
            tiles[k, :] = (x, y, hTile, wTile)
            k += 1
            if (j < (nTilesY-1)):
                y = y + hTile - remaindersY[j]
        if (i < (nTilesX-1)):
            x = x + wTile - remaindersX[i]

    return tiles

hTile = 256
wTile = 256
crop_dir_img = '../data/vaihingen-trial-data-crop/image'
crop_dir_label = '../data/vaihingen-trial-data-crop/label'
os.makedirs(crop_dir_img,exist_ok=True)
os.makedirs(crop_dir_label,exist_ok=True)

raw_dir = '../data/vaihingen-trial-data-raw'
raw_dir_img = os.path.join(raw_dir,'image')
raw_dir_label = os.path.join(raw_dir,'label')

for fname in tqdm(os.listdir(raw_dir_img)):
    img_path = os.path.join(raw_dir_img,fname)
    img = np.asarray(Image.open(img_path))
    (h,w,c) = img.shape

    label_path = os.path.join(raw_dir_label,fname)
    label = np.asarray(Image.open(label_path))

    tiles = crop_tiles(h,w,hTile,wTile)

    for i in range(tiles.shape[0]):
        coords = tiles[i]

        sample_img = img[coords[1]:(coords[1]+coords[3]),coords[0]:(coords[0]+coords[2]),:]
        sample_label = label[coords[1]:(coords[1]+coords[3]),coords[0]:(coords[0]+coords[2]),:]

        sample_img = Image.fromarray(sample_img)
        sample_label = Image.fromarray(sample_label)

        sample_img.save(os.path.join(crop_dir_img,fname.replace('.png','_'+str(i)+'.png')))
        sample_label.save(os.path.join(crop_dir_label,fname.replace('.png','_'+str(i)+'.png')))

