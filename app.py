"""
Gradio demo for anime frame interpolation.

Upload two consecutive anime frames, get the interpolated middle frame.

Run:
    python app.py
    python app.py --unet-checkpoint checkpoints/unet_best.pth
"""

import argparse
import os
import sys

import numpy as np
import torch
import gradio as gr
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.pipeline import AnimeInterpPipeline
from src.utils.edge import compute_distance_map


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(
    checkpoint_path: str,
    unet_checkpoint: str,
    device: torch.device,
) -> AnimeInterpPipeline:
    model = AnimeInterpPipeline(
        checkpoint_path=checkpoint_path,
        device=device,
    ).to(device)

    if unet_checkpoint and os.path.exists(unet_checkpoint):
        ckpt = torch.load(unet_checkpoint, map_location=device)
        model.unet.load_state_dict(ckpt["unet_state_dict"])
        print(f"Loaded fine-tuned Unet from {unet_checkpoint}")
    else:
        print("Using baseline IFNet only (no fine-tuned Unet)")

    model.eval()
    return model


def prepare_frame(
    pil_image: Image.Image,
    size: tuple,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert PIL image to model input tensors."""
    img = pil_image.convert("RGB").resize(
        (size[1], size[0]), Image.LANCZOS
    )
    arr = np.array(img)
    dist = compute_distance_map(arr)

    frame_t = torch.from_numpy(
        arr.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    dist_t = torch.from_numpy(dist).unsqueeze(0).unsqueeze(0).to(device)

    return frame_t, dist_t


def interpolate(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    model: AnimeInterpPipeline,
    device: torch.device,
    size: tuple = (256, 256),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gradio inference function.

    Args:
        frame_a: numpy array from Gradio image input
        frame_b: numpy array from Gradio image input

    Returns:
        tuple of (frame_a, interpolated, frame_b) for side-by-side display
    """
    if frame_a is None or frame_b is None:
        return None, None, None

    pil_a = Image.fromarray(frame_a.astype(np.uint8))
    pil_b = Image.fromarray(frame_b.astype(np.uint8))

    img0, dist_a = prepare_frame(pil_a, size, device)
    img1, dist_b = prepare_frame(pil_b, size, device)

    with torch.no_grad():
        pred = model(img0, img1, dist_a, dist_b)

    pred_np = (pred[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    # Also resize inputs for consistent display
    a_display = np.array(pil_a.resize((size[1], size[0])))
    b_display = np.array(pil_b.resize((size[1], size[0])))

    return a_display, pred_np, b_display


def build_demo(model, device, size):
    with gr.Blocks(title="Anime Frame Interpolation") as demo:
        gr.Markdown("""
        # Anime Frame Interpolation
        Upload two consecutive anime frames. The model predicts the missing middle frame.
        Fine-tuned on ATD-12K with distance-map guided refinement for anime flat colour regions.
        """)

        with gr.Row():
            input_a = gr.Image(label="Frame A (t=0)", type="numpy")
            input_b = gr.Image(label="Frame B (t=1)", type="numpy")

        btn = gr.Button("Interpolate", variant="primary")

        with gr.Row():
            out_a      = gr.Image(label="Frame A")
            out_interp = gr.Image(label="Interpolated (t=0.5)")
            out_b      = gr.Image(label="Frame B")

        btn.click(
            fn=lambda a, b: interpolate(a, b, model, device, size),
            inputs=[input_a, input_b],
            outputs=[out_a, out_interp, out_b],
        )

        gr.Markdown("""
        ### How it works
        1. IFNet estimates optical flow between Frame A and Frame B
        2. Frames are warped toward t=0.5 using the estimated flow
        3. UnetWithDistance refines the result using distance transform maps
           that explicitly identify flat colour regions prone to colour bleeding
        """)

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",      default="checkpoints/flownet.pkl")
    parser.add_argument("--unet-checkpoint", default="checkpoints/unet_best.pth")
    parser.add_argument("--size",            type=int, nargs=2,
                        default=[256, 256], metavar=("H", "W"))
    parser.add_argument("--share",           action="store_true",
                        help="Create public Gradio link")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model = load_model(args.checkpoint, args.unet_checkpoint, device)
    demo  = build_demo(model, device, tuple(args.size))

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()