import os
import numpy as np
import cv2
from config import tile_size, overlap, data_root, mask_root

def crop_tiles(image, mask, tile_size=tile_size, overlap=overlap):
    H, W = image.shape[:2]
    stride = tile_size - overlap
    tiles = []
    mask_tiles = []

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1 = min(y + tile_size, H)
            x1 = min(x + tile_size, W)

            tile = image[y:y1, x:x1]
            mask_tile = mask[y:y1, x:x1]

            pad_bottom = tile_size - (y1 - y)
            pad_right = tile_size - (x1 - x)
            if pad_bottom > 0 or pad_right > 0:
                tile = np.pad(tile, ((0,pad_bottom),(0,pad_right),(0,0)), mode='constant')
                mask_tile = np.pad(mask_tile, ((0,pad_bottom),(0,pad_right)), mode='constant')

            tiles.append(tile)
            mask_tiles.append(mask_tile)

    return tiles, mask_tiles

def main():
    os.makedirs('./data/tiles/images', exist_ok=True)
    os.makedirs('./data/tiles/masks', exist_ok=True)

    image_files = sorted(os.listdir(data_root))
    mask_files = sorted(os.listdir(mask_root))

    for i, (img_name, mask_name) in enumerate(zip(image_files, mask_files)):
        img_path = os.path.join(data_root, img_name)
        mask_path = os.path.join(mask_root, mask_name)

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        tiles, mask_tiles = crop_tiles(image, mask)

        for j, (t, m) in enumerate(zip(tiles, mask_tiles)):
            cv2.imwrite(f'./data/tiles/images/{i}_{j}.png', t)
            cv2.imwrite(f'./data/tiles/masks/{i}_{j}.png', m)

if __name__ == "__main__":
    main()
