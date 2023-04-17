from pathlib import Path
from PIL import Image
import numpy as np

# Search for pyNeRFRenderCore in build/Debug/
import sys
path_to_PyTurboNeRF = Path(__file__).parent.parent / "build" / "Debug"

print("Searching for TurboNeRF in", path_to_PyTurboNeRF)

sys.path.append(str(path_to_PyTurboNeRF))

import PyTurboNeRF as tn # type: ignore

# check if TurboNeRF is loaded
print("TurboNeRF loaded:", tn is not None)

# initialize all the things

manager = tn.Manager()

dataset = tn.Dataset("E:\\2022\\nerf-library\\testdata\\lego\\transforms.json")
dataset.load_transforms()

nerf = manager.create(dataset)

trainer = tn.Trainer(nerf, batch_size=2<<21)

renderer = tn.Renderer(pattern=tn.RenderPattern.LinearChunks)

# you can use any kind of render buffer you want, but if you want to get access to the rgba data as a np.array, you need to use the CPURenderBuffer
render_buf = tn.CPURenderBuffer()
render_buf.set_size(512, 512)
principal_point = (render_buf.width / 2, render_buf.height / 2)
focal_len = (500, 500)
shift = (0, 0)

# Just pull a random camera from the dataset
cam6 = dataset.cameras[6]

# Create a render camera with the resolution of our render buffer
render_cam = tn.Camera(
    (render_buf.width, render_buf.height),
    cam6.near,
    cam6.far,
    focal_len,
    principal_point,
    shift,
    cam6.transform,
    cam6.dist_params
)

# nerf2 = manager.load("H:\\dozer.turbo")
# request = tn.RenderRequest(
#     render_cam,
#     [nerf2],
#     render_buf,
#     tn.RenderModifiers(),
#     tn.RenderFlags.Final
# )

# renderer.submit(request)

# # save
# rgba = np.array(render_buf.get_rgba())
# rgba_uint8 = (rgba * 255).astype(np.uint8)
# img = Image.fromarray(rgba_uint8, mode="RGBA")
# img.save(f"H:\\render_loaded.png")

# exit()

# this method loads all the images and other data the Trainer needs


trainer.prepare_for_training()

def img_load_status(i, n):
    print(f"Loaded image {i} of {n}")

trainer.load_images(on_image_loaded=img_load_status)

for i in range(16):
    print(f"Training step {i}...")
    trainer.train_step()

    if i % 16 == 0 and i > 0:
        trainer.update_occupancy_grid(i)

    if i % 64 == 0 and i > 0:

        request = tn.RenderRequest(
            render_cam,
            [nerf],
            render_buf,
            tn.RenderModifiers(),
            tn.RenderFlags.Final
        )

        renderer.submit(request)

        # save
        rgba = np.array(render_buf.get_rgba())
        rgba_uint8 = (rgba * 255).astype(np.uint8)
        img = Image.fromarray(rgba_uint8, mode="RGBA")
        img.save(f"H:\\render_{i:05d}.png")

        # if you don't need direct access to the rgba data, you can use a CUDARenderBuffer (or a CPURenderBuffer) and the save_image method.
        # it is likely that this method is slightly faster than using numpy for just saving an image.
        # render_buf.save_image(f"H:\\render_{i:05d}.png")

        print(f"Saved render_{i:05d}.png!")

manager.save(nerf, "H:\\dozer.turbo")

# it is recommended to call these methods at the end of your program
render_buf.free()
tn.teardown()
