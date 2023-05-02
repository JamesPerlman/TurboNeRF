# RealityCapture -> NeRF transforms.json
# thank you GPT-4

import json
import numpy as np
import xml.etree.ElementTree as ET
import re

from argparse import ArgumentParser
from pathlib import Path
from PIL import Image

DEFAULT_NEAR = 0.05
DEFAULT_FAR = 128.0
DEFAULT_AABB = 16

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the input project (RealityCapture)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to the output file (JSON)")

    return parser.parse_args()


def get_image_dims(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size

def xmp_to_frame(xmp_data: str, img_dims: tuple[int, int], img_path_str: str) -> dict:
    img_width, img_height = img_dims

    # Parse XML data
    root = ET.fromstring(xmp_data)

    # Find the RDF description element
    rdf_ns = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}'
    xcr_ns = '{http://www.capturingreality.com/ns/xcr/1.1#}'
    description = root.find('rdf:RDF/rdf:Description', {'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'})


    # Extract values from XML
    fl_35mm = float(description.get(f'{xcr_ns}FocalLength35mm'))
    fl = fl_35mm / 36.0 * img_width
    cx = float(description.get(f'{xcr_ns}PrincipalPointU'))
    cy = float(description.get(f'{xcr_ns}PrincipalPointV'))

    distortion_coefs = list(map(float, description.find(f'{xcr_ns}DistortionCoeficients').text.split()))
    k1, k2, k3, p1, p2, _ = distortion_coefs

    
    rotation = list(map(float, description.find(f'{xcr_ns}Rotation').text.split()))
    position: list

    # sometimes xcr:Position is in the description tag
    position_str = description.find('xcr:Position', {'xcr': xcr_ns}).text if description.find('xcr:Position', {'xcr': xcr_ns}) is not None else description.get(f'{xcr_ns}Position')

    if position_str is None:
        position = list(map(float, description.find(f'{xcr_ns}Position').text.split()))
    else:
        position = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", position_str)))

    
    # Construct the transform matrix
    rotation_3x3 = np.array([
        rotation[:3],
        rotation[3:6],
        rotation[6:],
    ])

    # Transpose
    rotation_3x3 = rotation_3x3.T

    # Flip
    rotation_3x3[:, 1:3] *= -1

    # Construct the 4x4 matrix
    transform_4x4 = np.eye(4)
    transform_4x4[:3, :3] = rotation_3x3
    transform_4x4[:3, 3] = position

    # Construct the dictionary
    result = {
        "cx": (cx + 0.5) * img_width,
        "cy": (cy + 0.5) * img_height,
        "far": DEFAULT_FAR,
        "file_path": img_path_str,
        "fl_x": fl,
        "fl_y": fl,
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "near": DEFAULT_NEAR,
        "p1": p1,
        "p2": p2,
        "transform_matrix": transform_4x4.tolist(),
        "w": img_width,
        "h": img_height
    }

    return result

def read_file(file_path: Path) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_json(file_path: Path, data: dict) -> None:
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def get_relative_or_absolute_path(target_path, base_path):
    try:
        relative_path = target_path.relative_to(base_path)
    except ValueError:
        relative_path = target_path.absolute()
    return relative_path

if __name__ == '__main__':
    args = parse_args()

    output_path = Path(args.output)

    input_path = Path(args.input)

    path_to_imgs_and_xmps = input_path

    # Group images and xmps
    xmp_paths: list[Path] = [path for path in path_to_imgs_and_xmps.iterdir() if path.suffix == ".xmp"]
    img_paths: list[Path] = [path for path in path_to_imgs_and_xmps.iterdir() if path.stem in [xmp.stem for xmp in xmp_paths] and path.suffix != ".xmp"]

    imgs_and_xmps = list(zip(img_paths, xmp_paths))

    frames = []
    for img_path, xmp_path in imgs_and_xmps:
        
        relative_img_path = get_relative_or_absolute_path(img_path, output_path)
        xmp_data = read_file(xmp_path)
        img_dims = get_image_dims(img_path)
        frame = xmp_to_frame(xmp_data, img_dims, str(relative_img_path))
        frames.append(frame)
    

    transforms_data = {
        "aabb_scale": DEFAULT_AABB,
        "frames": frames
    }

    # if all frames have the same w and h, move w and h into the root
    if all([frame["w"] == frames[0]["w"] and frame["h"] == frames[0]["h"] for frame in frames]):
        transforms_data["w"] = frames[0]["w"]
        transforms_data["h"] = frames[0]["h"]

    write_json(output_path, transforms_data)
