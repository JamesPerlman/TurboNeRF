from pathlib import Path

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

nerf = manager.create_trainable(dataset.bounding_box)

trainer = tn.Trainer(dataset, nerf, batch_size=2<<21)

renderer = tn.Renderer()

render_buf = tn.CUDARenderBuffer()
render_buf.set_size(512, 512)

# Just pull a random camera from the dataset
cam6 = dataset.cameras[6]

# Create a render camera with the resolution of our render buffer
render_cam = tn.Camera(
    (render_buf.width, render_buf.height),
    cam6.near,
    cam6.far,
    cam6.focal_length,
    cam6.view_angle,
    cam6.transform,
    cam6.dist_params
)

# this method loads all the images and other data the Trainer needs
trainer.prepare_for_training()

for i in range(1024):
    print(f"Training step {i}...")
    trainer.train_step()

    if i % 16 == 0 and i > 0:
        trainer.update_occupancy_grid(i)

    if i % 32 == 0 and i > 0:

        request = tn.RenderRequest(
            render_cam,
            [nerf],
            render_buf
        )

        renderer.submit(request)
        render_buf.save_image(f"H:\\render_{i:05d}.png")
        print(f"Saved render_{i:05d}.png!")

# it is recommended to call these methods at the end of your program
render_buf.free()
tn.teardown()
