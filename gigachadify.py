from pathlib import Path
import torch


GIGACHAD_IMAGE_PATH = "TODO"


def detect_faces(image: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def face_detection_asserts(base_faces, target_faces):
    raise NotImplementedError


def get_face_single(faces: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def swap_faces(
    base_face_img: torch.Tensor, target_face_img: torch.Tensor
) -> torch.Tensor:
    base_faces = detect_faces(base_face_img)
    target_faces = detect_faces(target_face_img)

    face_detection_asserts(base_faces, target_faces)

    face_swapper = get_face_swap_model()

    base_face = get_face_single(base_faces)
    target_face = get_face_single(target_faces)

    result = target_face_img
    result = face_swapper.get(result, base_face, target_face)


def read_image(image_path: str):
    raise NotImplementedError


def save_image(image: torch.tensor, output_image_path: str):
    raise NotImplementedError


def get_face_swap_model():
    pass


def gigachadify(input_image_path: str, output_image_path: str):
    chad_base = read_image(GIGACHAD_IMAGE_PATH)
    input_image = read_image(input_image_path)

    result = swap_faces(chad_base, input_image)

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
