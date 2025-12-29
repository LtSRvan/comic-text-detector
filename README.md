This repository contains training scripts to train a text detector based on [manga-image-translator](https://github.com/zyddnys/manga-image-translator) which can extract bounding-boxes, text lines and segmentation of text from manga or comics to help further comics translation procedures such as text-removal, recognition, lettering, etc.  

There are some awesome projects such as manga-image-translator, [manga_ocr](https://github.com/kha-white/manga_ocr), [SickZil-Machine](https://github.com/KUR-creative/SickZil-Machine) offer DL models to automize the remaining work, <s>we are working on a computer-aided comic/manga translation software which would (hopefully) put them together.</s>  see [BallonsTranslator](https://github.com/dmMaze/BallonsTranslator)[WIP]

Download the text detection model from https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.2.1 or [Google Drive](https://drive.google.com/drive/folders/1cTsXP5NYTCjhPVxwScdhxqJleHuIOyXG?usp=sharing). 

# Examples

![AisazuNihaIrarenai-003](data/doc/AisazuNihaIrarenai-003.jpg)
<sup>(source: [manga109](http://www.manga109.org/en/), © Yoshi Masako)</sup>

![AisazuNihaIrarenai-003-mask](data/doc/AisazuNihaIrarenai-003-mask.png)

![AisazuNihaIrarenai-003-bboxes](data/doc/AisazuNihaIrarenai-003-bboxes.jpg)

# Training Details

Our current model can be summarized as below.  

<img src='data/doc/model.jpg'>  

All models were trained on around 13 thousand anime & comic style images, 1/3 from Manga109-s, 1/3 from [DCM](https://digitalcomicmuseum.com/), and 1/3 are synthetic data in a weak supervision manner due to the lack of available high-quality annotations.   

We used text detection model of manga-image-translator to generate text lines annotations for manga, and [Manga-Text-Segmentation](https://github.com/juvian/Manga-Text-Segmentation) with some post-processing to generate masks for both manga and comics. Synthetic data were generated using around 4k text-free anime-girls pictures from https://t.me/SugarPic, text-rendering, Unet and DBNet training scripts can be found in this repo.  Text block detector was trained using [yolov5 official repository](https://github.com/ultralytics/yolov5)  

We would not (don't have the right) share training sets or fonts publicly, 2/3 of the training set is not so clean anyway, so the training is reproducible only if you have enough images and fonts, you can use the models this repo provided to generate labels for comics/manga, and the comic style text rendering script to generate synthetic data, please refer to [examples.ipynb](examples.ipynb) for more details. 

## Acknowledgements

* [https://github.com/zyddnys/manga-image-translator](https://github.com/zyddnys/manga-image-translator)
* [https://github.com/juvian/Manga-Text-Segmentation](https://github.com/juvian/Manga-Text-Segmentation)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/WenmuZhou/DBNet.pytorch](https://github.com/WenmuZhou/DBNet.pytorch)

---

## Recent Updates

This fork includes the following changes to modernize the codebase:

- **Updated deprecated numpy types**: Replaced deprecated funtions in numpy 2.0+ to compatible alternatives in `utils/io_utils.py`.

- **Removed ONNX support**: Eliminated ONNX export/import functionality (`utils/export.py`, `TextDetBaseDNN` class) due to multiple issues with onnx simplifier. The detector now uses PyTorch only.

- **GPU by default**: The detector now automatically uses GPU (`cuda`) when available. Set `device='cpu'` explicitly if needed.

- **Simplified detection modes**: Added `mode` parameter with two options:
  - `mask` (default): Returns segmentation masks for text areas
  - `bbox`: Returns detection bounding boxes
  - Use `TextDetector(..., mode='bbox')` or CLI arg `--mode bbox`

- **Auto-download model**: If the model file is not found at `data/comictextdetector.pt`, it will be automatically downloaded from the official release.

- **CLI interface**: Added argparse support for easy use from command line:
  ```bash
  python inference.py -i <input_dir_or_file> [-o <output_dir>] [--mode mask|bbox] [--device cuda|cpu] [--half]
  ```

**Note:** Training scripts were not modified. All training-related code remains unchanged.
