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

manager = tn.NeRFManager()

renderer = tn.Renderer(pattern=tn.RenderPattern.LinearChunks)


LAYER_N = 3

def posix_to_win(path):
    win_path_1 = Path(path).as_posix().replace("/", "\\")
    #replace first part with drive letter
    win_path_2 = win_path_1.replace("\\e\\", "E:\\", 1)
    return win_path_2
# big-brain-mala-up-high/  double-antler-cones/         lumpy-short/  short-chandelier-on-spiny-bed/  tall-single-chandelier/          tiny-yellow-on-slab/
# crown-of-thorns/         lumpy-rocky-tall-parker-bg/  lumpy-solo/   red-urchin-thick-spines/  stalagmite-lumpy/               thick-wide-chandelier-on-rocks/  tiny-yellow-up-high/
# 
nerf_names = [
    "big-brain-mala-up-high",
    "double-antler-cones",
    "lumpy-short",
    "short-chandelier-on-spiny-bed",
    "tall-single-chandelier",
    "tiny-yellow-on-slab",
    "crown-of-thorns",
    "lumpy-rocky-tall-parker-bg",
    "lumpy-solo",
    "red-urchin-thick-spines",
    "stalagmite-lumpy",
    "thick-wide-chandelier-on-rocks",
    "tiny-yellow-up-high"
]

# alphabetize
nerf_names.sort()

# get CUDA_VISIBLE_DEVICES
import os
id = os.environ["CUDA_VISIBLE_DEVICES"]

# split into 4 equal parts
nerf_names = np.array_split(nerf_names, 4)[int(id)]

# prepend /e/nerfs/coral
nerf_names = [f"/e/nerfs/coral/{name}" for name in nerf_names]

# convert to windows path
nerf_names = [posix_to_win(name) for name in nerf_names]

# get index from args
index = int(sys.argv[1])

if index >= len(nerf_names):
    print(f"Index {index} is too high, there are only {len(nerf_names)} nerfs")
    exit()

base_path = nerf_names[index]


print(f"Working on {base_path}")

# create paths
Path(f"{base_path}\\test").mkdir(parents=True, exist_ok=True)
Path(f"{base_path}\\snapshots").mkdir(parents=True, exist_ok=True)

dataset = tn.Dataset(f"{base_path}\\video.transforms.json")
dataset.load_transforms()

nerf = manager.create()
nerf.attach_dataset(dataset)

trainer = tn.Trainer(nerf)

# you can use any kind of render buffer you want, but if you want to get access to the rgba data as a np.array, you need to use the CPURenderBuffer
render_buf = tn.CPURenderBuffer()
render_buf.set_size(512, 512)
principal_point = (render_buf.width / 2, render_buf.height / 2)
focal_len = (500, 500)
shift = (0, 0)

# Just pull a random camera from the dataset
cam0 = dataset.cameras[0]

# Create a render camera with the resolution of our render buffer
render_cam = tn.Camera(
    (render_buf.width, render_buf.height),
    cam0.near,
    cam0.far,
    focal_len,
    principal_point,
    shift,
    cam0.transform,
    cam0.dist_params
)

trainer.setup_data(batch_size=2<<21)

def img_load_status(i, n):
    print(f"Loaded image {i} of {n}")

trainer.load_images(on_image_loaded=img_load_status)

for i in range(15000):
    print(f"Training step {i}...")
    trainer.train_step()

    if i % 16 == 0 and i > 0:
        trainer.update_occupancy_grid(i)
    
    # render output image
    if i % 5000 == 0 and i > 0:

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
        img.save(f"{base_path}\\test\\render_{i:05d}.png")

    # save snapshot
    # if (i + 1) % 5000 == 0 and i > 0:
    #    tn.FileManager.save(nerf, f"{base_path}\\snapshots\\step-{nerf.training_step}.turbo")

tn.FileManager.save(nerf, f"{base_path}\\snapshots\\step-{nerf.training_step}.turbo")
trainer.teardown()
manager.destroy(nerf)

# it is recommended to call these methods at the end of your program
# render_buf.free()
# tn.teardown()
