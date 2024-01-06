#!/usr/bin/env python3.9

import os
import logging
import gradio as gr
import pandas as pd

import torch

from inference_single import get_videofact_model, load_single_video, process_single_video
from utils import *
from rich.logging import RichHandler
from typing import *

DEV_MODE = False

VideoFACT_xfer = None
VideoFACT_df = None

videofact_df_threshold = 0.33
videofact_xfer_threshold = 0.4

output_root_dir = "output"
application_name = "VideoFACT"
version = "0.1.0"

logger = logging.getLogger("videofact_gui")
logger.setLevel(logging.INFO)
handler = RichHandler()
log_fmt = "[%(filename)s:%(funcName)s] %(message)s"
handler.setFormatter(logging.Formatter(log_fmt))
logger.addHandler(handler)


@torch.no_grad()
def video_forgery_detection(
    video_path: str,
    shuffle,
    max_num_samples,
    sample_every,
    batch_size,
    num_workers,
    progress=gr.Progress(track_tqdm=True),
) -> List[str]:
    if video_path is None:
        raise ValueError("video_path cannot be None")

    global VideoFACT_xfer
    if VideoFACT_xfer is None:
        VideoFACT_xfer = get_videofact_model("xfer")

    dataloader = load_single_video(
        video_path,
        shuffle,
        int(max_num_samples),
        int(sample_every),
        int(batch_size),
        int(num_workers),
    )
    results = process_single_video(VideoFACT_xfer, dataloader, progress=progress)
    result_frame_paths, idxs, scores = list(zip(*results))
    decisions = ["Forged" if score > videofact_xfer_threshold else "Authentic" for score in scores]
    detection_graph = pd.DataFrame(
        {
            "frame": idxs,
            "score": scores,
            "decision": decisions,
        }
    )

    return (
        result_frame_paths,
        list(zip(idxs, scores)),
        f"Frame: {idxs[0]}, Score: {scores[0]:.5f}",
        "Forged" if scores[0] > videofact_xfer_threshold else "Authentic",
        gr.ScatterPlot.update(
            detection_graph,
            x="frame",
            y="score",
            color="decision",
            title="Frame Level Forgery Detection Score",
            tooltip=["frame", "score"],
            x_title="Frame",
            y_title="Score",
            x_lim=(-1, detection_graph["frame"].max() + 5),
            y_lim=(0, 1.05),
            interactive=True,
        ),
    )


@torch.no_grad()
def video_deepfake_detection(
    video_path: str,
    shuffle,
    max_num_samples,
    sample_every,
    batch_size,
    num_workers,
    progress=gr.Progress(track_tqdm=True),
) -> List[str]:
    if video_path is None:
        raise ValueError("video_path cannot be None")

    global VideoFACT_df
    if VideoFACT_df is None:
        VideoFACT_df = get_videofact_model("df")

    dataloader = load_single_video(
        video_path,
        shuffle,
        int(max_num_samples),
        int(sample_every),
        int(batch_size),
        int(num_workers),
    )
    results = process_single_video(VideoFACT_df, dataloader, progress=progress)
    result_frame_paths, idxs, scores = list(zip(*results))
    decisions = ["Deepfaked" if score > videofact_df_threshold else "Authentic" for score in scores]
    detection_graph = pd.DataFrame(
        {
            "frame": idxs,
            "score": scores,
            "decision": decisions,
        }
    )

    return (
        result_frame_paths,
        list(zip(idxs, scores)),
        f"Frame: {idxs[0]}, Score: {scores[0]:.5f}",
        "Deepfaked" if scores[0] > videofact_df_threshold else "Authentic",
        gr.ScatterPlot.update(
            detection_graph,
            x="frame",
            y="score",
            color="decision",
            title="Frame Level Deepfake Detection Score",
            tooltip=["frame", "score"],
            x_title="Frame",
            y_title="Score",
            x_lim=(0, detection_graph["frame"].max()),
            y_lim=(0, 1.05),
            interactive=True,
        ),
    )


css = """
:root {
    --det_score_height: 90px;
    --output_height: calc(35vh - var(--det_score_height));

    --c1_det_score_height: var(--det_score_height);
    --c1_img_output_height: var(--output_height);

    --c2_score_distributions_height: 300px;

    --c3_det_score_height: var(--det_score_height);
    --c3_output_gallery: var(--output_height);

    --c4_det_score_height: var(--det_score_height);
    --c4_output_gallery: var(--output_height);
}
#page-title {
    font-size: 2rem;
}
/* Styling for Image Forgery Detection */
#c1_det_score > label:nth-child(2) > textarea:nth-child(2),
#c1_det_decision > label:nth-child(2) > textarea:nth-child(2) {
    resize: none;
    height: var(--c1_det_score_height);
}
#c1_img_output {
    height: var(--c1_img_output_height) !important;
}
#c1_img_input {
    height: 100% !important;
}
#c1_img_input > div:nth-child(1), #c1_img_input > div:nth-child(3) {
    height: calc(var(--c1_img_output_height) + var(--c1_det_score_height) + 2vh + 2.0rem);
}
#c1_auth_notice > p, #c3_auth_notice > p {
    font-weight: bold;
    color: yellow;
    background-color: var(--color-grey-700);
    padding-left: 1rem;
}

/* Styling for Synthetic Image Detection */
#c2_score_distributions, #c2_score_distributions > * > * > * > canvas {
    height: var(--c2_score_distributions_height) !important;
}
#c2_img_input {
    height: 100% !important;
}
#c2_img_input > div:nth-child(1), #c2_img_input > div:nth-child(3) {
    height: calc(var(--c2_score_distributions_height) + 140px + 2vh + 2.2rem);
}

/* Styling for Video Forgery Detection */
#c3_det_score > label:nth-child(2) > textarea:nth-child(2),
#c3_det_decision > label:nth-child(2) > textarea:nth-child(2) {
    resize: none;
    height: var(--c3_det_score_height);
}
#c3_output_gallery {
    height: var(--c3_output_gallery) !important;
}
#c3_output_gallery > div.preview, #c3_output_gallery > div.grid-wrap {
    min-height: auto !important;
}
#c3_detection_graph, #c3_detection_graph > * > * > * > canvas {
    height: var(--c3_output_gallery) !important;
}
#c3_video_input {
    height: 100% !important;
}
#c3_video_input > .wrap {
    height: calc(var(--c3_output_gallery) * 2 + var(--c3_det_score_height) + 2vh + 3.0rem) !important;
}
#c3_auth_notice > p {
    font-weight: bold;
    color: yellow;
    background-color: var(--color-grey-700);
    padding-left: 1rem;
}

/* Styling for Video Deepfake Detection */
#c4_det_score > label:nth-child(2) > textarea:nth-child(2),
#c4_det_decision > label:nth-child(2) > textarea:nth-child(2) {
    resize: none;
    height: var(--c4_det_score_height);
}
#c4_output_gallery {
    height: var(--c4_output_gallery) !important;
}
#c4_output_gallery > div.preview, #c4_output_gallery > div.grid-wrap {
    min-height: auto !important;
}
#c4_detection_graph, #c4_detection_graph > * > * > * > canvas {
    height: var(--c4_output_gallery) !important;
}
#c4_video_input {
    height: 100% !important;
}
#c4_video_input > .wrap {
    height: calc(var(--c4_output_gallery) * 2 + var(--c4_det_score_height) + 2vh + 3.0rem) !important;
}
#c4_auth_notice > p {
    font-weight: bold;
    color: yellow;
    background-color: var(--color-grey-700);
    padding-left: 1rem;
}
#c4_auth_notice > p {
    font-weight: bold;
    color: yellow;
    background-color: var(--color-grey-700);
    padding-left: 1rem;
}

.preview > .thumbnails {
    -ms-overflow-style: auto !important;
    scrollbar-width: auto !important;
    justify-content: normal !important;
    padding-left: 7px !important;
}
.preview > .thumbnails::-webkit-scrollbar {
    display: block !important;
}
.preview > .thumbnails::-webkit-scrollbar-track {
    background-color: var(--color-grey-700) !important;
}
.preview > .thumbnails::-webkit-scrollbar-thumb {
    background-color: var(--color-grey-500) !important;
}
"""

app_name = f"{application_name} GUI - Version {version}"
with gr.Blocks(title=app_name, css=css) as demo:
    # Define the UI components
    gr.Markdown(app_name, elem_id="page-title")

    ########## Video Forgery Detection ##########
    with gr.Tab("Video Forgery Detection"):
        with gr.Row():
            with gr.Column():
                c3_vid_input = gr.Video(source="upload", interactive=True, elem_id="c3_video_input")
            with gr.Column():
                c3_det_scores = gr.State([])
                with gr.Row():
                    c3_det_score = gr.Textbox(
                        label="Detection Score", interactive=False, elem_id="c3_det_score"
                    )
                    c3_det_decision = gr.Textbox(
                        label="Decision", interactive=False, elem_id="c3_det_decision"
                    )
                gr.Markdown(
                    "NOTICE: Localization Map is not meaningful if the frame is authentic.",
                    elem_id="c3_auth_notice",
                )
                c3_gallery_output = gr.Gallery(
                    label="Predicted_Output",
                    interactive=False,
                    elem_id="c3_output_gallery",
                )
                c3_detection_graph = gr.ScatterPlot(label="", elem_id="c3_detection_graph")
        c3_submit_btn = gr.Button("Analyze")
        with gr.Accordion(label="Run Options", open=False):
            c3_opt_max_num_samples = gr.Number(value=100, label="max_num_samples", interactive=True)
            c3_opt_sample_every = gr.Number(value=5, label="sample_every", interactive=True)
            c3_opt_shuffle = gr.Checkbox(value=True, label="shuffle", interactive=True)
            c3_opt_batch_size = gr.Number(value=1, label="batch_size", interactive=False)
            c3_opt_num_worker = gr.Number(value=8, label="num_worker", interactive=False)
        with gr.Row():
            c3_examples = gr.Examples(
                list_dir("examples/xfer"),
                inputs=[c3_vid_input],
            )

    ########## Deepfake Detection ##########
    with gr.Tab("Deepfake Detection"):
        with gr.Row():
            with gr.Column():
                c4_vid_input = gr.Video(source="upload", interactive=True, elem_id="c4_video_input")
            with gr.Column():
                c4_det_scores = gr.State([])
                with gr.Row():
                    c4_det_score = gr.Textbox(
                        label="Detection Score", interactive=False, elem_id="c4_det_score"
                    )
                    c4_det_decision = gr.Textbox(
                        label="Decision", interactive=False, elem_id="c4_det_decision"
                    )
                gr.Markdown(
                    "NOTICE: Localization Map is not meaningful if the frame is authentic.",
                    elem_id="c4_auth_notice",
                )
                c4_gallery_output = gr.Gallery(
                    label="Predicted_Output",
                    interactive=False,
                    elem_id="c4_output_gallery",
                )
                c4_detection_graph = gr.ScatterPlot(label="", elem_id="c4_detection_graph")
        c4_submit_btn = gr.Button("Analyze")
        with gr.Accordion(label="Run Options", open=False):
            c4_opt_max_num_samples = gr.Number(value=100, label="max_num_samples", interactive=True)
            c4_opt_sample_every = gr.Number(value=5, label="sample_every", interactive=True)
            c4_opt_shuffle = gr.Checkbox(value=True, label="shuffle", interactive=True)
            c4_opt_batch_size = gr.Number(value=2, label="batch_size", interactive=False)
            c4_opt_num_worker = gr.Number(value=5, label="num_worker", interactive=False)
        with gr.Row():
            c4_examples = gr.Examples(
                list_dir("examples/df"),
                inputs=[c4_vid_input],
            )

    def c3_on_gallery_select(event: gr.SelectData, scores):
        return (
            f"Frame: {scores[event.index][0]}, Score: {scores[event.index][1]:.5f}",
            "Forged" if scores[event.index][1] > videofact_xfer_threshold else "Authentic",
        )

    def c4_on_gallery_select(event: gr.SelectData, scores):
        return (
            f"Frame: {scores[event.index][0]}, Score: {scores[event.index][1]:.5f}",
            "Deepfaked" if scores[event.index][1] > videofact_df_threshold else "Authentic",
        )

    # Define the functionalities of the UI components
    ##### C3 #####
    c3_vid_input.upload(lambda v: v, [c3_vid_input], c3_vid_input)
    c3_gallery_output.select(
        fn=c3_on_gallery_select, inputs=[c3_det_scores], outputs=[c3_det_score, c3_det_decision]
    )
    c3_gallery_output.style(preview=True)
    c3_submit_btn.click(
        fn=video_forgery_detection,
        inputs=[
            c3_vid_input,
            c3_opt_shuffle,
            c3_opt_max_num_samples,
            c3_opt_sample_every,
            c3_opt_batch_size,
            c3_opt_num_worker,
        ],
        outputs=[
            c3_gallery_output,
            c3_det_scores,
            c3_det_score,
            c3_det_decision,
            c3_detection_graph,
        ],
    )
    ##### C4 #####
    c4_vid_input.upload(lambda v: v, [c4_vid_input], c4_vid_input)
    c4_gallery_output.select(
        fn=c4_on_gallery_select, inputs=[c4_det_scores], outputs=[c4_det_score, c4_det_decision]
    )
    c4_gallery_output.style(preview=True)
    c4_submit_btn.click(
        fn=video_deepfake_detection,
        inputs=[
            c4_vid_input,
            c4_opt_shuffle,
            c4_opt_max_num_samples,
            c4_opt_sample_every,
            c4_opt_batch_size,
            c4_opt_num_worker,
        ],
        outputs=[
            c4_gallery_output,
            c4_det_scores,
            c4_det_score,
            c4_det_decision,
            c4_detection_graph,
        ],
    )

if __name__ == "__main__":
    demo.queue().launch(share=False, show_error=True)
