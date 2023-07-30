from pathlib import Path
from PIL import Image
import numpy as np
import json

# Search for pyNeRFRenderCore in build/Debug/
import sys
path_to_PyTurboNeRF = Path(__file__).parent.parent / "build" / "Debug"

print("Searching for TurboNeRF in", path_to_PyTurboNeRF)

sys.path.append(str(path_to_PyTurboNeRF))

import PyTurboNeRF as tn # type: ignore

# check if TurboNeRF is loaded
print("TurboNeRF loaded:", tn is not None)

# initialize all the things

manager = tn.NeRFManager()

data_path = Path("D:/Dropbox/PictureNeRFect/source/boat")
frames_path = data_path / "frames"
frames_path.mkdir(exist_ok=True)
dataset = tn.Dataset(str(data_path / "transforms2.json"))
dataset.load_transforms()

nerf = manager.create(dataset)

trainer = tn.Trainer(nerf)

renderer = tn.Renderer(pattern=tn.RenderPattern.LinearChunks)

# you can use any kind of render buffer you want, but if you want to get access to the rgba data as a np.array, you need to use the CPURenderBuffer
render_buf = tn.CPURenderBuffer()

w = 512
h = 512

cams = []
with open(str(data_path / "render.json")) as f:
    data = json.load(f)
    w = int(data['w'])
    h = int(data['h'])
    frames = data['frames']
    for frame in frames:
        cam_json = frame['camera']
        transform = np.array(cam_json['transform'])
        near = cam_json['near']
        far = cam_json['far']
        focal_len = cam_json['focal_length']

        cam = tn.Camera(
            (w, h),
            near,
            far,
            (focal_len, focal_len),
            (w/2, h/2),
            (0, 0),
            tn.Transform4f(transform),
            tn.DistortionParams()
        )
        cams.append(cam)

render_buf.set_size(w, h)
trainer.setup_data(batch_size=2<<14)

def img_load_status(i, n):
    print(f"Loaded image {i} of {n}")

trainer.load_images(on_image_loaded=img_load_status)

step = 0

nc = len(cams)
accel_point = int(nc * 2 / 3)

print("accel_point", accel_point)
for c in range(len(cams)):

    for _ in range(2 * max(1, c - accel_point)):
        print(f"Training step {step}...")
        trainer.train_step()

        if step % 16 == 0 and step > 0:
            trainer.update_occupancy_grid(step)
        
        step += 1

    render_cam = cams[c]

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
    img.save(str(frames_path / f"render_{c:05d}.png"))

    # if you don't need direct access to the rgba data, you can use a CUDARenderBuffer (or a CPURenderBuffer) and the save_image method.
    # it is likely that this method is slightly faster than using numpy for just saving an image.
    # render_buf.save_image(f"H:\\render_{i:05d}.png")

    print(f"Saved render_{c:05d}.png!")

# it is recommended to call these methods at the end of your program
render_buf.free()
tn.teardown()
