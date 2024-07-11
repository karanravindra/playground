import os
from PIL import Image
import numpy as np


def load_images_from_folder(folder: str) -> list[Image.Image]:
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def average_color(image: Image.Image) -> np.ndarray:
    np_image = np.array(image)
    w, h, d = np_image.shape
    return np.mean(np_image.reshape(w * h, d), axis=0)


def find_closest_image(target_avg: np.ndarray, avgs: list[np.ndarray]) -> int:
    avgs = np.array(avgs)
    idx = np.linalg.norm(avgs - target_avg, axis=1).argmin()
    return idx


def create_image_mosaic(
    target_image_path: str, tiles_folder: str, tile_size: tuple[int, int] = (50, 50)
) -> Image.Image:
    target_image = Image.open(target_image_path)
    tiles = load_images_from_folder(tiles_folder)
    tile_avgs = [average_color(tile.resize(tile_size)) for tile in tiles]

    target_w, target_h = target_image.size
    target_w = (target_w // tile_size[0]) * tile_size[0]
    target_h = (target_h // tile_size[1]) * tile_size[1]
    target_image = target_image.resize((target_w, target_h))

    mosaic = Image.new("RGB", (target_w, target_h))
    for i in range(0, target_w, tile_size[0]):
        for j in range(0, target_h, tile_size[1]):
            sub_image = target_image.crop((i, j, i + tile_size[0], j + tile_size[1]))
            avg_color = average_color(sub_image)
            closest_idx = find_closest_image(avg_color, tile_avgs)
            tile = tiles[closest_idx].resize(tile_size)
            mosaic.paste(tile, (i, j))

    return mosaic


# Example usage:
if __name__ == "__main__":
    target_image_path = "path/to/target/image.jpg"
    tiles_folder = "path/to/tiles/folder"
    mosaic = create_image_mosaic(target_image_path, tiles_folder)
    mosaic.save("path/to/save/mosaic.jpg")
