import sys
import argparse
import torch

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.watermark.image_watermark.raw import RAWProcessor
from modules.watermark.image_watermark.pipeline import image_watermark_pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["add_watermark", "detect", "evaluate"],
        help="Task to perform."
    )

    # subparser: Define different parameters for different image watermark methods
    subparsers = parser.add_subparsers(dest="algorithm", required=True)

    # RAW algorithm
    raw_parser = subparsers.add_parser("RAW", help="RAW watermark parameters")
    raw_parser.add_argument('--media_type', type=str, choices=['image', 'video'], default='video', help='Choose media type: "image" or "video".')
    raw_parser.add_argument('--image_path', type=str, default='../data/dataset/amber/image/AMBER_1.jpg', help='Path to input image file for watermarking.')
    raw_parser.add_argument('--video_dir', type=str, default='../modules/watermark/image_watermark/utils/raw/video_examples', help='directory containing input videos.')
    raw_parser.add_argument("--output_dir", default="output", type=str)
    raw_parser.add_argument('--crop_size', action='store_true', help='Whether to crop video frames to 512x512.')
    raw_parser.add_argument('--injection_every_k_frames', type=int, default=1, help='Insert watermark every k frames.')
    raw_parser.add_argument('--decision_thres', type=float, default=0.5, help='Threshold for watermark detection.')
    raw_parser.add_argument('--fps', type=int, default=30, help='Frames per second for output video.')
    raw_parser.add_argument('--num_frames', type=int, default=25, help='Number of frames to process.')
    raw_parser.add_argument('--wm_index', type=int, default=0, help='Watermark index (from 0-3).')
    raw_parser.add_argument('--save', action='store_true', help='Save the output (images/videos) with watermark.')
    raw_parser.add_argument('--wt_image_path', type=str, default='', help='Path to wt image file for watermarking evaluation.')
    raw_parser.add_argument('--metric', type=str, default='PSNR', help='Metrics for watermarking evaluation,for PSNR\SSIM\LPIPS\VIF\JPEG-C.....')
    raw_parser.add_argument('--video_dir_wt', type=str, default='', help='directory containing watermark videos.')

    args = parser.parse_args()
    image_watermark_pipeline(args)