# -*- coding: UTF-8 -*-
import copy
from pathlib import Path

import torch

from yolov5_face.models.experimental import attempt_load  # noqa: E402
from yolov5_face.models.yolo import Model  # noqa: E402
from yolov5_face.utils.image import letterbox  # noqa: E402
from yolov5_face.utils.general import (  # noqa: E402
    check_img_size,
    non_max_suppression_face,
    scale_coords,
)


def load_model(
    weights, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """Load standard YOLO weights or fall back to a modern checkpoint layout."""
    weight_path = Path(weights)
    if weight_path.is_file() and weight_path.suffix == ".pt":
        ckpt = torch.load(weight_path, map_location=device)
        if {"state_dict", "model_cfg"} <= ckpt.keys():
            cfg = ckpt["model_cfg"]
            model = Model(cfg, ch=cfg.get("ch", 3), nc=cfg.get("nc", 1))
            model.load_state_dict(ckpt["state_dict"], strict=False)
            if "stride" in ckpt:
                model.stride = ckpt["stride"]
            if "names" in ckpt:
                model.names = ckpt["names"]
            return model.float().eval().requires_grad_(False).to(device)
    # fallback to original behavior (handles lists, URLs, etc.)
    return attempt_load(weights, map_location=device)


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2,
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    # clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords


def detect_landmarks(
    model, im, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    img_size = im.shape[0]
    conf_thres = 0.6
    iou_thres = 0.5

    img0 = copy.deepcopy(im)
    imgsz = check_img_size(img_size, s=model.stride.max())
    img = letterbox(img0, new_shape=imgsz)[0]
    img = img.transpose(2, 0, 1).copy()

    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]

    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    results = []
    for det in pred:
        im0 = img0.copy()

        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            det[:, 5:15] = scale_coords_landmarks(
                img.shape[2:], det[:, 5:15], im0.shape
            ).round()

            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = det[j, 5:15].view(-1).tolist()
                class_num = det[j, 15].cpu().numpy()

                results.append(
                    {
                        "box": xyxy,
                        "confidence": conf,
                        "landmarks": landmarks,
                        "class": class_num,
                    }
                )

    return results
