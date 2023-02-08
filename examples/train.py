from pathlib import Path

# Search for pyNeRFRenderCore in build/Debug/
import sys
path_to_pyNeRFRenderCore = Path(__file__).parent.parent / "build" / "Debug"

print("Searching for PyNeRFRenderCore in", path_to_pyNeRFRenderCore)

sys.path.append(str(path_to_pyNeRFRenderCore))

import PyNeRFRenderCore as nrc # type: ignore

# check if PyNeRFRenderCore is loaded
print("PyNeRFRenderCore loaded:", nrc is not None)

# initialize all the things

manager = nrc.Manager()

dataset = nrc.Dataset("E:\\2022\\nerf-library\\testdata\\lego\\transforms.json")

nerf = manager.create_trainable(dataset.bounding_box)

trainer = nrc.Trainer(dataset, nerf, batch_size=2<<21)

renderer = nrc.Renderer(batch_size=2<<20)

render_buf = nrc.RenderBuffer(512, 512)

# Just pull a random camera from the dataset
cam6 = dataset.cameras[6]

# Create a render camera with the resolution of our render buffer
render_cam = nrc.Camera(
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
        trainer.update_occupancy_grid(0.9)

    if i % 32 == 0 and i > 0:
        req = nrc.RenderRequest(render_cam, [nerf], render_buf)
        renderer.request(req)
        render_buf.save_image(f"H:\\render_{i:05d}.png")
        print(f"Saved render_{i:05d}.png!")

# it is recommended to call this method at the end of your program

nrc.teardown()
