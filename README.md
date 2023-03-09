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
git clone git@github.com:JamesPerlman/NeRFRenderCore --recursive
cd NeRFRenderCore
cmake . -B build
cmake --build build -j
```

Until we have an extensible data loader, the test data I'm working with is here:  

https://www.dropbox.com/sh/qkt4t1tk1o7pdc6/AAD218LLtAavRZykYl33mO8ia?dl=1

This project has only been tested on that one Lego scene.  Real scenes appear to be broken for now.  

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


## CITATIONS

Next-level respect to the researchers much of this codebase is based off.  Thank you for making your research public.  This would not have been possible without you.  

Mildenhall, Ben, et al. "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." arXiv, 2020.  doi:10.48550/arxiv.2003.08934 - (https://arxiv.org/abs/2003.08934)  
Müller, Thomas, et al. "Instant neural graphics primitives with a multiresolution hash encoding." *ACM Trans. Graph.*, 41(4), 102:1-102:15 - https://doi.org/10.1145/3528223.3530127  
Max, Nelson. "Optical Models for Direct Volume Rendering." IEEE Transactions on Visualization and Computer Graphics (1995) - https://courses.cs.duke.edu/spring03/cps296.8/papers/max95opticalModelsForDirectVolumeRendering.pdf  
Müller, T. (2021). tiny-cuda-nn (Version 1.7) [Computer software]. https://github.com/NVlabs/tiny-cuda-nn  
Fawzi, A., Balog, M., Huang, A. et al. Discovering faster matrix multiplication algorithms with reinforcement learning. Nature 610, 47–53 (2022). https://doi.org/10.1038/s41586-022-05172-4  
Alman, Josh, and Virginia Vassilevska Williams. "A Refined Laser Method and Faster Matrix Multiplication." arXiv, 2020, doi:10.48550/arxiv.2010.05846.  https://arxiv.org/abs/2010.05846  
Sabour, Sara, et al. "RobustNeRF: Ignoring Distractors with Robust Losses." arXiv, 2023, arXiv:2302.00833.  https://arxiv.org/abs/2302.00833  

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
