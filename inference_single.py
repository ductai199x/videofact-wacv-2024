import os
import cv2
import decord
import random
import tempfile
import yaml
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from gradio import Progress
from typing import *

from model.common.videofact import VideoFACT as VideoFACT_Module
from model.videofact_pl_wrapper import VideoFACTPLWrapper


xfer_ckpt = "weights/videofact_xfer.ckpt"
df_ckpt = "weights/videofact_df.ckpt"

default_config = yaml.full_load(open("configs/default.yaml"))


def get_videofact_model(type: Literal["xfer", "df"] = "xfer"):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if type == "xfer":
        model = VideoFACTPLWrapper.load_from_checkpoint(
            xfer_ckpt, model=VideoFACT_Module, map_location=device, **default_config
        )
    elif type == "df":
        model = VideoFACTPLWrapper.load_from_checkpoint(
            df_ckpt, model=VideoFACT_Module, map_location=device, **default_config
        )
    else:
        raise ValueError(f"Unknown type: {type}")

    model = model.to(device)
    model = model.eval()
    return model


@torch.no_grad()
def process_single_video(
    model,
    dataloader: DataLoader,
    output_dir: Union[str, None] = None,
    progress: Union[Progress, None] = None,
) -> Tuple[List[Tuple[str, float, int]]]:
    class_preds = []
    frame_idxs = []
    result_frames = []
    if output_dir is None or not os.path.exists(output_dir) or not os.path.isdir(output_dir):
        output_dir = tempfile.mkdtemp()
    if isinstance(progress, Progress):
        iterable = progress.tqdm(dataloader)
    else:
        iterable = tqdm(dataloader, desc="Processing frames", position=1, leave=False)
    for batch in iterable:
        frames, idxs = batch
        class_out, patch_out = model(frames.to(model.device))
        class_out, patch_out = class_out.detach().cpu(), patch_out.detach().cpu()
        class_out = torch.softmax(class_out, dim=1)
        patch_preds = [
            model.patch_to_pixel_pred(
                pl,
                model.patch_size,
                model.img_size,
                min_thresh=0.1,
                max_num_regions=3,
                final_thresh=0.30,
            )
            for pl in patch_out
        ]
        pixel_preds = torch.vstack([pp.get_pixel_preds().unsqueeze(0) for pp in patch_preds])
        frame_idxs.append(idxs)
        class_preds.append(class_out[:, 1])

        for idx, frame, pixel_pred in zip(idxs, frames, pixel_preds):
            result_frame = frame.permute(1, 2, 0).numpy() / 255.0
            pixel_pred = pixel_pred.unsqueeze(-1).repeat(1, 1, 3).numpy()
            result_frame = cv2.addWeighted(result_frame, 1 - 0.3, pixel_pred, 0.3, 0)
            resize_to = int(result_frame.shape[1] * 0.70), int(result_frame.shape[0] * 0.70)
            result_frame = cv2.resize(result_frame, resize_to)
            cv2.imwrite(
                f"{output_dir}/{idx}.jpg",
                cv2.cvtColor(result_frame * 255, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 40],
            )
            result_frames.append(os.path.abspath(f"{output_dir}/{idx}.jpg"))
    class_preds = torch.concat(class_preds).tolist()
    frame_idxs = torch.concat(frame_idxs).tolist()

    results = list(zip(result_frames, frame_idxs, class_preds))
    results = sorted(results, key=lambda x: x[1])

    return results


def load_single_video(
    video_path,
    shuffle,
    max_num_samples,
    sample_every,
    batch_size,
    num_workers,
):
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video_path, num_threads=8)
    batch_idxs = list(range(0, len(vr), sample_every))
    if shuffle:
        random.shuffle(batch_idxs)
    if max_num_samples > 0:
        batch_idxs = batch_idxs[:max_num_samples]

    batch_idxs = sorted(batch_idxs)
    frame_batch = vr.get_batch(batch_idxs)
    frame_batch = frame_batch.permute(0, 3, 1, 2).float()
    if frame_batch.shape[2] > frame_batch.shape[3]:
        frame_batch = frame_batch.permute(0, 1, 3, 2)
        frame_batch = torchvision.transforms.functional.vflip(frame_batch)
    if frame_batch.shape[2] != 1080 or frame_batch.shape[3] != 1920:
        frame_batch = torchvision.transforms.functional.resize(frame_batch, (1080, 1920), antialias=True)

    return DataLoader(
        list(zip(frame_batch, batch_idxs)),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
