# Image Mosaic Generator

This Python script generates an image mosaic by arranging multiple smaller images (tiles) to closely mimic a target image. This process involves matching the average color of each tile with segments of the target image to recreate the overall appearance as a mosaic.

## Features

- **Average Color Matching:** Tiles are selected and placed based on the closest average color match to the corresponding segment of the target image.
- **Custom Tile Sizes:** Users can define the size of the tiles to control the granularity of the mosaic.
- **Flexible Image Input:** Any folder of images can be used as a source of tiles, and any image can be set as the target.

## Prerequisites

Ensure you have Python installed on your system. The script depends on the following Python libraries:

- `Pillow` (PIL Fork) for image handling
- `numpy` for numerical operations

You can install these dependencies via pip:

```bash
pip install Pillow numpy
```

## Usage

1. **Prepare Your Images:**
   - Place all tile images in a single folder. These images will be used to construct the mosaic.
   - Choose your target image, which the mosaic will mimic.

2. **Run the Script:**
   - Modify the script to point to your target image and the folder containing your tiles.
   - You can adjust the tile size by changing the `tile_size` parameter in the `create_image_mosaic` function call.

   Example usage in the script:

   ```python
   mosaic = create_image_mosaic('path/to/target/image.jpg', 'path/to/tiles/folder')
   mosaic.show()  # To view the mosaic
   mosaic.save('path/to/save/mosaic.jpg')  # To save the mosaic
   ```
