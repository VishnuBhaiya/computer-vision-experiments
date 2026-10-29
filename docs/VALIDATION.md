# Release validation

Checked on 29 October 2026 in an isolated publishing copy.

- Three notebook schemas and transformed Python syntax checked: 29 handwritten-symbol code cells, 41 dog-breed code cells and 35 object-detection code cells.
- The notebook's actual Sobel feature implementation compared with an independent vectorised NumPy reference, including a constant-image boundary check.
- Embedded ClearML credentials removed; offline tracking is selected when personal credentials are absent.
- Dataset roots made configurable through CV_SYMBOLS_DATA, CV_DOGS_DATA, CV_PASCAL_DATA and CV_YOLO_DATA.
- Public sources scanned for credentials; saved outputs and account/widget metadata cleared.

Not performed: fresh SVM/ANN/ResNet/Faster R-CNN/YOLO training or full dataset evaluation. Dataset distributions, original splits and GPU training must be supplied separately. No fresh accuracy or mAP score is claimed by a syntax/feature smoke test.
