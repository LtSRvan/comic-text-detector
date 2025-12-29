import json
import argparse
import os
import sys
from basemodel import TextDetBase, DET_MODE_MASK, DET_MODE_BBOX
import os.path as osp
from tqdm import tqdm
import numpy as np
import cv2
import torch
from pathlib import Path
from utils.yolov5_utils import non_max_suppression
from utils.db_utils import SegDetectorRepresenter
from utils.io_utils import imread, imwrite, find_all_imgs, NumpyEncoder
from utils.imgproc_utils import letterbox, xyxy2yolo, get_yololabel_strings
from utils.textblock import TextBlock, group_output, visualize_textblocks
from utils.textmask import refine_mask, refine_undetected_mask, REFINEMASK_INPAINT, REFINEMASK_ANNOTATION
from utils.model_utils import get_model_path, DEFAULT_MODEL_PATH
from pathlib import Path
from typing import Union

def model2annotations(model_path, img_dir_list, save_dir, save_json=False, device=None):
    if isinstance(img_dir_list, str):
        img_dir_list = [img_dir_list]
    model = TextDetector(model_path=model_path, input_size=1024, device=device, act='leaky')
    imglist = []
    for img_dir in img_dir_list:
        imglist += find_all_imgs(img_dir, abs_path=True)
    for img_path in tqdm(imglist):
        imgname = osp.basename(img_path)
        img = imread(img_path)
        im_h, im_w = img.shape[:2]
        imname = imgname.replace(Path(imgname).suffix, '')
        maskname = 'mask-'+imname+'.png'
        poly_save_path = osp.join(save_dir, 'line-' + imname + '.txt')
        mask, mask_refined, blk_list = model(img, refine_mode=REFINEMASK_ANNOTATION, keep_undetected_mask=True)
        polys = []
        blk_xyxy = []
        blk_dict_list = []
        for blk in blk_list:
            polys += blk.lines
            blk_xyxy.append(blk.xyxy)
            blk_dict_list.append(blk.to_dict())
        blk_xyxy = xyxy2yolo(blk_xyxy, im_w, im_h)
        if blk_xyxy is not None:
            cls_list = [1] * len(blk_xyxy)
            yolo_label = get_yololabel_strings(cls_list, blk_xyxy)
        else:
            yolo_label = ''
        with open(osp.join(save_dir, imname+'.txt'), 'w', encoding='utf8') as f:
            f.write(yolo_label)

        if len(polys) != 0:
            if isinstance(polys, list):
                polys = np.array(polys)
            polys = polys.reshape(-1, 8)
            np.savetxt(poly_save_path, polys, fmt='%d')
        if save_json:
            with open(osp.join(save_dir, imname+'.json'), 'w', encoding='utf8') as f:
                f.write(json.dumps(blk_dict_list, ensure_ascii=False, cls=NumpyEncoder))
        imwrite(osp.join(save_dir, imgname), img)
        imwrite(osp.join(save_dir, maskname), mask_refined)

def preprocess_img(img, input_size=(1024, 1024), device='cpu', bgr2rgb=True, half=False, to_tensor=True):
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in, ratio, (dw, dh) = letterbox(img, new_shape=input_size, auto=False, stride=64)
    if to_tensor:
        img_in = img_in.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_in = np.array([np.ascontiguousarray(img_in)]).astype(np.float32) / 255
        if to_tensor:
            img_in = torch.from_numpy(img_in).to(device)
            if half:
                img_in = img_in.half()
    return img_in, ratio, int(dw), int(dh)

def postprocess_mask(img: Union[torch.Tensor, np.ndarray], thresh=None):
    if isinstance(img, torch.Tensor):
        img = img.squeeze_()
        if img.device != 'cpu':
            img = img.detach_().cpu()
        img = img.numpy()
    else:
        img = img.squeeze()
    if thresh is not None:
        img = img > thresh
    img = img * 255
    return img.astype(np.uint8)

def postprocess_yolo(det, conf_thresh, nms_thresh, resize_ratio, sort_func=None):
    det = non_max_suppression(det, conf_thresh, nms_thresh)[0]
    if det.device != 'cpu':
        det = det.detach_().cpu().numpy()
    det[..., [0, 2]] = det[..., [0, 2]] * resize_ratio[0]
    det[..., [1, 3]] = det[..., [1, 3]] * resize_ratio[1]
    if sort_func is not None:
        det = sort_func(det)

    blines = det[..., 0:4].astype(np.int32)
    confs = np.round(det[..., 4], 3)
    cls = det[..., 5].astype(np.int32)
    return blines, cls, confs

class TextDetector:
    lang_list = ['eng', 'ja', 'unknown']
    langcls2idx = {'eng': 0, 'ja': 1, 'unknown': 2}

    def __init__(self, model_path=None, input_size=1024, device=None, half=False,
                 nms_thresh=0.35, conf_thresh=0.4, mask_thresh=0.3, act='leaky',
                 mode=DET_MODE_MASK):
        super(TextDetector, self).__init__()
        model_path = get_model_path(model_path)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # ONNX removed to many stupid errors
        self.net = TextDetBase(model_path, device=device, act=act)
        self.backend = 'torch'

        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size
        self.half = half
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.seg_rep = SegDetectorRepresenter(thresh=0.3)
        self.mode = mode

    @torch.no_grad()
    def __call__(self, img, refine_mode=REFINEMASK_INPAINT, keep_undetected_mask=False):
        img_in, ratio, dw, dh = preprocess_img(img, input_size=self.input_size, device=self.device, half=self.half, to_tensor=True)
        im_h, im_w = img.shape[:2]

        blks, mask, lines_map = self.net(img_in)

        resize_ratio = (im_w / (self.input_size[0] - dw), im_h / (self.input_size[1] - dh))
        blks = postprocess_yolo(blks, self.conf_thresh, self.nms_thresh, resize_ratio)

        mask = postprocess_mask(mask)

        lines, scores = self.seg_rep(self.input_size, lines_map)
        box_thresh = 0.6
        idx = np.where(scores[0] > box_thresh)
        lines, scores = lines[0][idx], scores[0][idx]

        mask = mask[: mask.shape[0]-dh, : mask.shape[1]-dw]
        mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
        if lines.size == 0 :
            lines = []
        else :
            lines = lines.astype(np.float64)
            lines[..., 0] *= resize_ratio[0]
            lines[..., 1] *= resize_ratio[1]
            lines = lines.astype(np.int32)
        blk_list = group_output(blks, lines, im_w, im_h, mask)

        if self.mode == DET_MODE_MASK:
            # refined mask
            mask_refined = refine_mask(img, mask, blk_list, refine_mode=refine_mode)
            if keep_undetected_mask:
                mask_refined = refine_undetected_mask(img, mask, mask_refined, blk_list, refine_mode=refine_mode)
            return mask, mask_refined, blk_list
        else:
            # BBOXES
            return blks, mask, blk_list

def traverse_by_dict(img_dir_list, dict_dir):
    if isinstance(img_dir_list, str):
        img_dir_list = [img_dir_list]
    imglist = []
    for img_dir in img_dir_list:
        imglist += find_all_imgs(img_dir, abs_path=True)
    for img_path in tqdm(imglist):
        imgname = osp.basename(img_path)
        imname = imgname.replace(Path(imgname).suffix, '')
        mask_path = osp.join(dict_dir, 'mask-'+imname+'.png')
        with open(osp.join(dict_dir, imname+'.json'), 'r', encoding='utf8') as f:
            blk_dict_list = json.loads(f.read())
            blk_list = [TextBlock(**blk_dict) for blk_dict in blk_dict_list]
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = refine_mask(img, mask, blk_list)

        visualize_textblocks(img, blk_list)
        cv2.imshow('im', img)
        cv2.imshow('mask', mask)
        cv2.waitKey(0)

def parse_args():
    parser = argparse.ArgumentParser(description='Comic Text Detector - Detect text in manga/comics')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input image or directory containing images')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output directory (default: <input>/_output)')
    parser.add_argument('-m', '--model', type=str, default=None,
                        help=f'Path to model file (default: {DEFAULT_MODEL_PATH})')
    parser.add_argument('-d', '--device', type=str, default=None,
                        help='Device to use: cuda or cpu (default: cuda if available)')
    parser.add_argument('--mode', type=str, default=DET_MODE_MASK, choices=[DET_MODE_MASK, DET_MODE_BBOX],
                        help=f'Detection mode: mask (segmentation) or bbox (detection) (default: {DET_MODE_MASK})')
    parser.add_argument('--no-json', action='store_true',
                        help='Do not save JSON output')
    parser.add_argument('--half', action='store_true',
                        help='Use half precision (FP16) for faster inference')
    return parser.parse_args()

def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist")
        sys.exit(1)

    if args.output is None:
        if input_path.is_file():
            output_dir = input_path.parent / '_output'
        else:
            output_dir = input_path / '_output'
    else:
        output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    # This is so Stupid but more stupid is Mike
    device = args.device
    if device is not None and device.lower() not in ('cuda', 'cpu'):
        print(f"Error: Invalid device '{device}'. Use 'cuda' or 'cpu'")
        sys.exit(1)
    if device and device.lower() == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")

    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    print(f"Device: {device or ('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Mode:   {args.mode}")
    print("-" * 50)

    # Auto-model downloader if not in path
    detector = TextDetector(
        model_path=args.model,
        device=device,
        half=args.half,
        mode=args.mode
    )

    if input_path.is_file():
        img_dirs = [input_path.parent]
        img_names = [input_path.name]
    else:
        img_dirs = [input_path]
        img_names = None

    imglist = []
    for img_dir in img_dirs:
        imglist += find_all_imgs(img_dir, abs_path=True)

    if not imglist:
        print("No images found in input directory")
        sys.exit(1)

    print(f"Processing {len(imglist)} image(s)...")


    for img_path in tqdm(imglist):
        imgname = osp.basename(img_path)
        img = imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue

        im_h, im_w = img.shape[:2]
        imname = imgname.replace(Path(imgname).suffix, '')

        result1, result2, blk_list = detector(img, refine_mode=REFINEMASK_INPAINT, keep_undetected_mask=True)

        if args.mode == DET_MODE_MASK:
            mask_refined = result2
            maskname = 'mask-' + imname + '.png'
            # Save refined mask
            imwrite(osp.join(output_dir, maskname), mask_refined)

            # polygon data
            polys = []
            for blk in blk_list:
                polys += blk.lines
            if polys:
                polys = np.array(polys).reshape(-1, 8)
                poly_save_path = osp.join(output_dir, 'line-' + imname + '.txt')
                np.savetxt(poly_save_path, polys, fmt='%d')
        else:
            # bbox mode
            blks = result1
            blk_xyxy = [blk.xyxy for blk in blk_list]
            if blk_xyxy:
                blk_xyxy = np.array(blk_xyxy)
                for xyxy in blk_xyxy:
                    cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)

        if not args.no_json:
            blk_dict_list = [blk.to_dict() for blk in blk_list]
            with open(osp.join(output_dir, imname+'.json'), 'w', encoding='utf8') as f:
                f.write(json.dumps(blk_dict_list, ensure_ascii=False, cls=NumpyEncoder))

    print(f"\nDone! Output saved to: {output_dir}")

if __name__ == '__main__':
    main()
