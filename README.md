# Computer Vision Experiments

Three computer-vision investigations by Vishnu Bhaiya, organised as a source-only research portfolio.

| Experiment | Methods | Notebook |
| --- | --- | --- |
| Handwritten mathematical symbols | SVM / ANN; raw pixels, LBP, HOG and Sobel features; ClearML tracking | [Symbols](notebooks/handwritten-symbols.ipynb) |
| Ten-class dog-breed classification | Transfer learning with baseline and customised ResNet50 | [Dog breeds](notebooks/dog-breed-classification.ipynb) |
| Aerial maritime object detection | Faster R-CNN and YOLOv8s; split-level evaluation and failure inspection | [Detection](notebooks/aerial-maritime-detection.ipynb) |

## Run

Use a fresh Google Colab GPU runtime for the training notebooks, or create an isolated Python 3.12/3.13 environment and install `requirements.txt`. Read each notebook's configuration before running: configure the dataset roots and provide the expected folder/annotation layout. Personal Google Drive paths are not portable dataset downloads.

ClearML credentials are read from `CLEARML_API_ACCESS_KEY` and `CLEARML_API_SECRET_KEY`; no key is included. Create your own ClearML configuration if experiment tracking is used. Dataset distributions and trained models are excluded because redistribution rights for the datasets and experiment splits are not established here.

## Validation and evidence

Run `python scripts/validate_notebooks.py` for notebook schema and transformed-Python syntax checks. These checks do not train the networks, load missing datasets or assert fresh benchmark scores. The original notebooks had saved executions; portfolio outputs and account/widget metadata were cleared. Full GPU and dataset reproduction remains a separate step. See [validation](docs/VALIDATION.md).

Upstream model and dataset licences still apply. The experiments cover feature engineering, transfer learning and object detection. SolarVision is available as a separate application repository.
