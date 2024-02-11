import os
from pathlib import Path
from PIL import Image, ImageOps
import requests
from io import BytesIO
import matplotlib.pyplot as plt

import numpy as np
import torch
import insightface
import cv2


GIGACHAD_IMAGE_PATH = "images/gigachad.png"

MODELS_DIR = os.path.join(Path(__file__).parent, "models")
INSIGHTFACE_PATH = os.path.join(MODELS_DIR, "insightface")

if torch.cuda is not None:
    if torch.cuda.is_available():
        PROVIDER = ["CUDAExecutionProvider"]
    elif torch.backends.mps.is_available():
        PROVIDER = ["CoreMLExecutionProvider"]
    else:
        PROVIDER = ["CPUExecutionProvider"]


def get_face_analysis_model() -> insightface.app.FaceAnalysis:
    return insightface.app.FaceAnalysis(
        name="buffalo_l", providers=PROVIDER, root=INSIGHTFACE_PATH
    )


def detect_faces(
    img_data: np.ndarray,
    face_analyser: insightface.app.FaceAnalysis,
    det_size=(640, 640),
) -> list:
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser.get(img_data)


def face_detection_asserts(base_faces, target_faces):
    raise NotImplementedError


def get_face_single(faces: torch.tensor) -> torch.tensor:
    # TODO
    return faces[0]


def swap_faces(
    base_face_img: np.ndarray,
    target_face_img: np.ndarray,
    face_analysis_model: insightface.app.FaceAnalysis,
) -> torch.tensor:
    base_faces = detect_faces(base_face_img, face_analysis_model)
    target_faces = detect_faces(target_face_img, face_analysis_model)

    # face_detection_asserts(base_faces, target_faces)

    face_swapper = get_face_swap_model()

    base_face = get_face_single(base_faces)
    target_face = get_face_single(target_faces)

    result = base_face_img
    result = face_swapper.get(result, base_face, target_face)

    return result


def is_url(image_path: str) -> bool:
    """Check if the given path is a URL."""
    return image_path.startswith("http://") or image_path.startswith("https://")


def read_image(image_path: str) -> np.ndarray:
    if is_url(image_path):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = (image * 255).astype(np.uint8)
    return image


def save_image(image: np.ndarray, output_image_path: str):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image.save(output_image_path)


def get_face_swap_model():
    return insightface.model_zoo.get_model(
        os.path.join(INSIGHTFACE_PATH, "inswapper_128.onnx"), providers=PROVIDER
    )


def gigachadify(input_image_path: str, output_image_path: str):
    chad_base = read_image(GIGACHAD_IMAGE_PATH)
    input_image = read_image(input_image_path)

    face_analysis_model = get_face_analysis_model()

    result = swap_faces(chad_base, input_image, face_analysis_model)

    save_image(result, output_image_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gigachadify any face")

    parser.add_argument(
        "--input-image", type=str, required=True, help="Path or URL to the input image"
    )
    parser.add_argument(
        "--output-image",
        type=str,
        default="outputs/gigachad.png",
        help="Path to the output image",
    )

    args = parser.parse_args()

    gigachadify(args.input_image, args.output_image)
