# gigachadify

🗿 gigachadify any face: Put any face onto the Gigachad meme with this script.
This tool automatically detects a face in your image and swaps it into the
legendary Gigachad

"Giulio did you really make a script that swaps your face into Gigachad?"

![gigagiuliochad](outputs/gigagiuliochad.png)

"Yes."

## Setup

1. Clone this repo
2. Install the requirements outlined in [pyproject.toml](pyproject.toml)
3. Download the faceswap model from
   [huggingface](https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx)
   and put it in [models/insightface](models/insightface)

## Usage

```stdout
usage: python gigachadify.py [-h] --input-image INPUT_IMAGE
                      [--output-image OUTPUT_IMAGE]

Gigachadify any face

options:
  -h, --help            show this help message and exit
  --input-image INPUT_IMAGE
                        Path or URL to the input image
  --output-image OUTPUT_IMAGE
                        Path to the output image (default: outputs/gigachad.png)
```

## License

This project relies on
[insightface](https://github.com/deepinsight/insightface). Please refer to their
license.

This project also relies on
[Gourieff's ReActor](https://huggingface.co/Gourieff/ReActor) models. Please
refer to their license.

Otherwise, this project is licensed under the MIT License. See
[LICENSE](LICENSE)

## Project state

MVP. Quick and dirty, will polish in coming weeks.
