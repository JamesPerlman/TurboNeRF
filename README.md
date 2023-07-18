![TurboNeRF](https://github.com/slowcon/TurboNeRF/blob/main/assets/images/TurboNERF-GH-Logo.png?raw=true)
[![Discord](https://img.shields.io/discord/1083484809873064046?label=Discord&logo=Discord&logoColor=white)](https://discord.gg/pfRTqT2mvb)
## IN OPEN ALPHA 0.0.1!

Happy Pi Day! This project is now in open alpha testing.  For the full story, as well as instructions and support resources, please see this video and its description: https://youtu.be/TeWYAbhgaiU  

Enjoy!  


## PRE-RELEASE TO-DO LIST
- [x] Python API
- [x] Blender Bridge
- [ ] Save NeRF  
- [ ] Load NeRF  
- [ ] Multi-GPU  
- [ ] Multi-NeRF  
- [ ] Dataset Preparation Tools  
- [x] Downloadable Binaries  

Wishlist:
- [ ] Drastically improve rendering quality / reduce training loss  

## INTRODUCTION

Hello NeRF enthusiasts!  Here you will find my NeRF rendering and training library.  The core principles of this NeRF method are based on the incredible work of [Thomas Müller](https://tom94.net/), [Alex Evans](https://research.nvidia.com/person/alex-evans), [Christoph Schied](https://research.nvidia.com/person/christoph-schied), and [Alex Keller](https://research.nvidia.com/person/alex-keller), in their paper [Instant neural graphics primitives with a multiresolution hash encoding](https://arxiv.org/abs/2003.08934).  

Yes, I realize there is already a [CUDA implementation](https://github.com/nvlabs/instant-ngp), but I wanted to take a crack at reimplementing this myself for the challenge, and for artistic uses such as:  

* Spatial distortions  
* Multiple NeRFs in one scene  
* Multi-GPU capabilities  
* Shadertoy-style effects  
* Fractals  

Since everything here has been written from scratch*, this codebase is permissively licensed and commercial-use-friendly.  

(*with generous help from [NeRF](https://github.com/bmild/nerf), [nerfstudio](https://github.com/nerfstudio-project/nerfstudio), [NerfAcc](https://github.com/KAIR-BAIR/nerfacc), and completely built on the [tiny-cuda-nn](https://github.com/NVLabs/tiny-cuda-nn) backend.)  

Enjoy!  
-James  
https://twitter.com/jperldev

## INSTALLATION

Required toolkits:
* [CUDA 12](https://developer.nvidia.com/cuda-downloads)
* [CMake 3.25.2](https://cmake.org/download/)

Tested Configuration:
* [Windows 11](https://www.microsoft.com/software-download/windows11)
* [NVIDIA RTX A6000](https://www.nvidia.com/en-us/design-visualization/rtx-a6000/)

Build steps:

```
git clone git@github.com:JamesPerlman/TurboNeRF --recursive
cd TurboNeRF
cmake . -B build
cmake --build build -j
```

Until we have an extensible data loader, the test data I'm working with is here:  

https://www.dropbox.com/sh/qkt4t1tk1o7pdc6/AAD218LLtAavRZykYl33mO8ia?dl=1

This project has only been tested on that one Lego scene.  Real scenes appear to be broken for now.  

### LINUX ADDITIONAL STEP

There is an open [issue](https://github.com/pybind/pybind11/issues/4606) when using CUDA 12 and PyBind11
(the latter is used by TurboNeRF for the Python module). Currently, a patch manually needs to be applied
after checking out the TurboNeRF repository as shown above:

```
git clone git@github.com:JamesPerlman/TurboNeRF --recursive
cd TurboNeRF
# Apply patch
patch -p1 < ../patches/pybind11-cuda12.patch
# Build as usual
cmake . -B build
cmake --build build -j
```

## THANK YOU

Extreme gratitude to open source projects that will allow this project to reach its full potential (in order of integration date):

* [CUDA CMake Starter](https://github.com/pkestene/cuda-proj-tmpl)  
* [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)  
* [JSON for modern C++](https://github.com/nlohmann/json)  
* [stb](https://github.com/nothings/stb)  
* [NeRF](https://github.com/bmild/nerf)  
* [nerfstudio](https://github.com/nerfstudio-project/nerfstudio)  
* [NerfAcc](https://github.com/KAIR-BAIR/nerfacc)  
* [torch-ngp](https://github.com/ashawkey/torch-ngp)  
* [Nerfies](https://github.com/google/nerfies)  
* [glfw](https://github.com/glfw/glfw)  
* [pybind11](https://github.com/pybind/pybind11)  
* [RenderMan](https://github.com/prman-pixar/RenderManForBlender)
* [glad](https://github.com/Dav1dde/glad)  
* [cuda-cmake-github-actions](https://github.com/ptheywood/cuda-cmake-github-actions)  
* [pure-torch-ngp](https://github.com/cheind/pure-torch-ngp)
* [torch_efficient_distloss](https://github.com/sunset1995/torch_efficient_distloss)


## CITATIONS

Next-level respect to the researchers much of this codebase is based off.  Thank you for making your research public.  This would not have been possible without you.  

[NeRF: Neural Radiance Fields](https://www.matthewtancik.com/nerf)
```
@inproceedings{mildenhall2020nerf,
 title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
 author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
 year={2020},
 booktitle={ECCV},
}
```

[Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://nvlabs.github.io/instant-ngp/)
```
@article{mueller2022instant,
    author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
    title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    journal = {ACM Trans. Graph.},
    issue_date = {July 2022},
    volume = {41},
    number = {4},
    month = jul,
    year = {2022},
    pages = {102:1--102:15},
    articleno = {102},
    numpages = {15},
    url = {https://doi.org/10.1145/3528223.3530127},
    doi = {10.1145/3528223.3530127},
    publisher = {ACM},
    address = {New York, NY, USA}
}
```

[NeRF in the Wild](https://nerf-w.github.io/)
```
@inproceedings{martinbrualla2020nerfw,
    author = {Martin-Brualla, Ricardo
            and Radwan, Noha
            and Sajjadi, Mehdi S. M.
            and Barron, Jonathan T.
            and Dosovitskiy, Alexey
            and Duckworth, Daniel},
    title = {{NeRF in the Wild: Neural Radiance Fields for
            Unconstrained Photo Collections}},
    booktitle = {CVPR},
    year={2021}
}
```

[Mip-NeRF 360](https://jonbarron.info/mipnerf360/)
```
@article{barron2022mipnerf360,
    title={Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields},
    author={Jonathan T. Barron and Ben Mildenhall and 
            Dor Verbin and Pratul P. Srinivasan and Peter Hedman},
    journal={CVPR},
    year={2022}
}
```

[tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)  
```
@software{Muller_tiny-cuda-nn_2021,
    author = {Müller, Thomas},
    license = {BSD-3-Clause},
    month = apr,
    title = {{tiny-cuda-nn}},
    url = {https://github.com/NVlabs/tiny-cuda-nn},
    version = {1.7},
    year = {2021}
}
```

Max, Nelson. *Optical Models for Direct Volume Rendering.* IEEE Transactions on Visualization and Computer Graphics (1995) - https://courses.cs.duke.edu/spring03/cps296.8/papers/max95opticalModelsForDirectVolumeRendering.pdf  
Fawzi, A., Balog, M., Huang, A. et al. *Discovering faster matrix multiplication algorithms with reinforcement learning.* Nature 610, 47–53 (2022) - https://doi.org/10.1038/s41586-022-05172-4  
Alman, Josh, and Virginia Vassilevska Williams. *A Refined Laser Method and Faster Matrix Multiplication.* arXiv, 2020, doi:10.48550/arxiv.2010.05846 - https://arxiv.org/abs/2010.05846  
Sabour, Sara, et al. *RobustNeRF: Ignoring Distractors with Robust Losses.* arXiv, 2023, arXiv:2302.00833 - https://arxiv.org/abs/2302.00833  

## SUPPORTERS

Extreme thank yous to these subscribers on Twitch (https://twitch.tv/jperldev) who support this project's development!

madclawgonzo - Requested a haiku written by ChatGPT: "Madclawgonzo / Subscribing to your stream / Software project."  
anonymous - Requested to remain anonymous  
gusround - https://github.com/candidogustavo  
slowcon - "uncle slowcon is here with the 4090"  
likid_3 - <3  
cognitrol - Supporting cool work that helps the community explore technology  
dankmatrix - (pending message)  
seferidis - (pending message)  
memepp - (pending message)  
Dakren12 - (pending message)  
Relakin - (Confused)  
flouwr - (pending message)  
