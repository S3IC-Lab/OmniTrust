import os
import piq
import cv2
import torch
import numpy as np
from PIL import Image
import pathlib, itertools
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.utils as vutils
import torchvision.io as io
from ..image_watermark.utils.raw import raw_watermark, tools
from ..image_watermark.utils.raw.raw_watermark import RAWatermark


class RAWProcessor:
    def __init__(self, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.wm_index = args.wm_index
        self.decision_thres = args.decision_thres
        self.injection_every_k_frames = args.injection_every_k_frames
        self.fps = args.fps
        self.save = args.save
        self.model = RAWatermark(device=self.device, wm_index=self.wm_index)
        self.args = args

    def add_watermark_image(self, image_path, out_prefix="image"):
        # If image_path points to a directory, process all images in the directory.
        p = pathlib.Path(image_path)
        if p.is_dir():
            exts = ('.jpg','.jpeg','.png','.bmp','.gif')
            files = [x for x in p.iterdir() if x.suffix.lower() in exts]
            if not files: raise FileNotFoundError("No image found in directory: {image_path}")
            for f in files:
                self.add_watermark_image(str(f), out_prefix=f.stem)
            return None

        # If image_math points to a single image, process it directly.
        img = Image.open(image_path).convert('RGB')
        transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor()
        ])
        input_img = transform(img).unsqueeze(0).to(self.device)
        wm_img = self.model.encode(input_img)

        if self.save:
            output_dir = self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)

            orig_path = os.path.join(output_dir, f"{out_prefix}_original.png")
            wm_path = os.path.join(output_dir, f"{out_prefix}_wm.png")

            vutils.save_image(input_img, orig_path)
            vutils.save_image(wm_img, wm_path)
            print(f"Saved: {orig_path}, {wm_path}")

        return input_img, wm_img


    def add_watermark_video(self, video_dir, num_frames=25, crop_size=False, output_dir=None):
        output_dir = self.args.output_dir if output_dir is None else output_dir
        orig_dir = os.path.join(output_dir, "originals")
        wm_dir   = os.path.join(output_dir, "watermarked")
        os.makedirs(orig_dir, exist_ok=True)
        os.makedirs(wm_dir, exist_ok=True)

        dataset = tools.VideoDataset(
            root_dir=video_dir,
            crop_size=crop_size,
            no_of_frames=num_frames
        )

        wm_videos = []
        original_videos = []

        for i in range(len(dataset)):
            video = dataset[i].to(self.device)  # [T,C,H,W] in [0,1]
            wm_video = self.model.encode(video, injection_every_k_frames=self.injection_every_k_frames)

            original_videos.append(video)
            wm_videos.append(wm_video)

            if self.save:
                orig_path = os.path.join(orig_dir, f"original_{i}.mp4")
                wm_path   = os.path.join(wm_dir,   f"watermarked_{i}.mp4")
                self._save_video(video, orig_path)
                self._save_video(wm_video, wm_path)
                print(f"Saved: {orig_path}, {wm_path}")

        return original_videos, wm_videos


    def detect_video(self, video_dir, num_frames=25, crop_size=False):
        dataset = tools.VideoDataset(
            root_dir=video_dir,
            crop_size=crop_size,
            no_of_frames=num_frames
        ) # size (videosnum, frames, H, W, C)

        detections = []
        for i in range(len(dataset)):
            video = dataset[i].to(self.device) # (frames, H, W, C)
            print(f"[Detect Result] Video {i}: ", end="")
            detected = self.model.detect(video, decision_thres=self.decision_thres)
            detections.append(detected)
        return detections
    

    def detect_image(self, image_path):
        # If image_path points to a directory, process all images in the directory.
        p = pathlib.Path(image_path)
        if p.is_dir():
            exts = ('.jpg','.jpeg','.png','.bmp','.gif')
            files = [x for x in p.iterdir() if x.suffix.lower() in exts]
            if not files:
                raise FileNotFoundError("No image found in directory: {image_path}")
            results = {}
            for f in files:
                results[f.name] = self.detect_image(str(f))
            return results

        # If image_path points to a single image, process it directly.
        img = Image.open(image_path).convert('RGB')
        transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor()
        ])
        img_t = transform(img).to(self.device)
        video = img_t.unsqueeze(0).repeat(self.fps, 1, 1, 1)
        print(f"[Detect Result] {pathlib.Path(image_path).name}: ", end="")
        detected = self.model.detect(video, decision_thres=self.decision_thres)
        return detected


    def evaluate_image(self, image_path, wt_image_path, metric):
        """ if you use the add watermark code before, the image_path and wt_image_path should is ../../../examples/output (equal), 0-ori, 1-wt, ... ."""
        def jpeg_attack(image_pa, quality=50):
            img = Image.open(image_pa)
            img.save('temp_jpeg.jpg', quality=quality)
            dete_result = self.detect_image(image_path='temp_jpeg.jpg')
            attacked = Image.open('temp_jpeg.jpg').convert('RGB')
            return torch.tensor(np.array(attacked)).permute(2, 0, 1).unsqueeze(0).float(), dete_result
        
        def calculate_ber(original_wm, extracted_wm): #BER
            # original_wm, extracted_wm is bin values numpy (QR code)
            return np.mean(original_wm != extracted_wm)
            # wm_ori = np.array(Image.open('wm_binary.png').convert('L')) > 128
            # wm_ext = np.array(Image.open('extracted_wm.png').convert('L')) > 128
            # print(f"BER: {calculate_ber(wm_ori, wm_ext):.2%}")

        transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor()
        ])

        folder_path_src = image_path  
        images_src = []  
        image_names_src = []
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')  
        image_files_src = [f for f in os.listdir(folder_path_src) if f.endswith(supported_formats)]
        for image_file in image_files_src:
            file_name_without_ext = os.path.splitext(image_file)[0]  
            image_names_src.append(file_name_without_ext)  
            file_path = os.path.join(folder_path_src, image_file)
            img = Image.open(file_path).convert('RGB')
            img = transform(img)
            images_src.append(img)  

        folder_path_wt = wt_image_path  
        images_wt = []  
        image_names_wt = []
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')  
        image_files_wt = [f for f in os.listdir(folder_path_wt) if f.endswith(supported_formats)]
        for image_file in image_files_wt:
            file_name_without_ext = os.path.splitext(image_file)[0]  
            image_names_wt.append(file_name_without_ext)  
            file_path = os.path.join(folder_path_wt, image_file)
            img = Image.open(file_path).convert('RGB')
            img = transform(img)
            images_wt.append(img)  

        psnrs = []
        ssims = []
        lpipss = []
        vifs = []
        jpegcs = []
        if image_path == wt_image_path:
            for i in range(min(len(images_src), len(images_wt))//2): 
                j = i*2
                k = i*2+1
                ori_t = torch.tensor(np.array(images_src[j])).permute(2, 0, 1).unsqueeze(0).float()
                wt_t = torch.tensor(np.array(images_wt[k])).permute(2, 0, 1).unsqueeze(0).float()
                if metric == 'PSNR': # quality
                    psnr = piq.psnr(ori_t, wt_t, data_range=255.)
                    psnrs.append(psnr)
                    # print(f"PSNR: {psnr:.2f} dB")  # >30 dB is great
                elif metric == 'SSIM':
                    ori_t = torch.tensor(np.array(images_src[j])).unsqueeze(0).float()
                    wt_t = torch.tensor(np.array(images_wt[k])).unsqueeze(0).float()
                    ssim = piq.ssim(ori_t, wt_t, data_range=255.)
                    # print(f"SSIM: {ssim:.4f}")  # [0,1], 1 is best
                    ssims.append(ssim)
                elif metric == 'LPIPS':
                    ori_t = torch.tensor(np.array(images_src[j])).unsqueeze(0).float()
                    wt_t = torch.tensor(np.array(images_wt[k])).unsqueeze(0).float()
                    lpips = piq.LPIPS(reduction='none')
                    score = lpips(ori_t / 255., wt_t / 255.)
                    # print(f"LPIPS: {score.item():.4f}")  # [0,1]，0 is best
                    lpipss.append(score)
                elif metric == 'VIF':
                    ori_t = torch.tensor(np.array(images_src[j])).unsqueeze(0).float()
                    wt_t = torch.tensor(np.array(images_wt[k])).unsqueeze(0).float()
                    vif = piq.vif_p(ori_t / 255., wt_t / 255.)
                    # print(f"VIF: {vif:.4f}")  # [0,1]，1 is best
                    vifs.append(vif)
                elif metric == 'JPEG-C': # robustness
                    # JPEG compression
                    wt_attacked, dete_result = jpeg_attack(image_pa=os.path.join(wt_image_path, image_files_wt[i]), quality=30)
                    ori_t = torch.tensor(np.array(images_src[j])).unsqueeze(0).float()
                    wt_t = torch.tensor(np.array(images_wt[k])).unsqueeze(0).float()
                    psnr_attacked = piq.psnr(ori_t, wt_attacked, data_range=255.)
                    jpegcs.append(dete_result) # watermark is or not exist
                    jpegcs.append(psnr_attacked)
                    # print(f"PSNR after JPEG: {psnr_attacked:.2f} dB")
                else:
                    raise ValueError(f"Unsupported task type: {metric}")
        else:
            for i in range(min(len(images_src), len(images_wt))): 
                ori_t = torch.tensor(np.array(images_src[i])).permute(2, 0, 1).unsqueeze(0).float()
                wt_t = torch.tensor(np.array(images_wt[i])).permute(2, 0, 1).unsqueeze(0).float()
                if metric == 'PSNR': # quality
                    psnr = piq.psnr(ori_t, wt_t, data_range=255.)
                    psnrs.append(psnr)
                    # print(f"PSNR: {psnr:.2f} dB")  # >30 dB is great
                elif metric == 'SSIM':
                    ori_t = torch.tensor(np.array(images_src[j])).unsqueeze(0).float()
                    wt_t = torch.tensor(np.array(images_wt[k])).unsqueeze(0).float()
                    ssim = piq.ssim(ori_t, wt_t, data_range=255.)
                    # print(f"SSIM: {ssim:.4f}")  # [0,1], 1 is best
                    ssims.append(ssim)
                elif metric == 'LPIPS':
                    ori_t = torch.tensor(np.array(images_src[j])).unsqueeze(0).float()
                    wt_t = torch.tensor(np.array(images_wt[k])).unsqueeze(0).float()
                    lpips = piq.LPIPS(reduction='none')
                    score = lpips(ori_t / 255., wt_t / 255.)
                    # print(f"LPIPS: {score.item():.4f}")  # [0,1]，0 is best
                    lpipss.append(lpips)
                elif metric == 'VIF':
                    ori_t = torch.tensor(np.array(images_src[j])).unsqueeze(0).float()
                    wt_t = torch.tensor(np.array(images_wt[k])).unsqueeze(0).float()
                    vif = piq.vif_p(ori_t / 255., wt_t / 255.)
                    # print(f"VIF: {vif:.4f}")  # [0,1]，1 is best
                    vifs.append(vif)
                elif metric == 'JPEG-C': # robustness
                    # JPEG compression
                    wt_attacked, dete_result = jpeg_attack(image_pa=os.path.join(wt_image_path, image_files_wt[i]), quality=30)
                    ori_t = torch.tensor(np.array(images_src[j])).unsqueeze(0).float()
                    wt_t = torch.tensor(np.array(images_wt[k])).unsqueeze(0).float()
                    psnr_attacked = piq.psnr(ori_t, wt_attacked, data_range=255.)
                    jpegcs.append(dete_result) # watermark is or not exist
                    jpegcs.append(psnr_attacked)
                    # print(f"PSNR after JPEG: {psnr_attacked:.2f} dB")
                else:
                    raise ValueError(f"Unsupported task type: {metric}")

        if metric == 'PSNR':
            return print('the all psnrs is (should > 30dB):',psnrs)
        elif metric == 'SSIM':
            return print('the all ssims is (should closer 1):',ssims)
        elif metric == 'LPIPS':
            return print('the all lpips is (should closer 0):',lpipss)
        elif metric == 'VIF':
            return print('the all vif is (should closer 1):',vifs)
        elif metric == 'JPEG-C':
            return print(jpegcs[0], 'the jpeg compression after psnr is (should > 30dB) :',jpegcs[1])
        else:
            raise ValueError(f"Unsupported task type: {metric}")


    def evaluate_video(self, video_dir_ori, video_dir_wt, num_frames=25, crop_size=False, metric=None):
        dataset_ori = VideoDataset(
            root_dir=video_dir_ori,
            crop_size=crop_size,
            no_of_frames=num_frames
        ) # size (videosnum, frames, H, W, C)

        dataset_wt = VideoDataset(
            root_dir=video_dir_wt,
            crop_size=crop_size,
            no_of_frames=num_frames
        ) # size (videosnum, frames, H, W, C)

        def save_frames_pil(dataset, output_dir="video_frames"):
            os.makedirs(output_dir, exist_ok=True)
            video_list = []
            for video_idx in range(len(dataset)):
                frames = dataset[video_idx]  # numpy: [T, H, W, 3]
                if frames.size == 0:
                    raise ValueError(f"Video {video_idx} has no frames after decoding/cropping.")

                video_dir = os.path.join(output_dir, f"video_{video_idx}")
                os.makedirs(video_dir, exist_ok=True)
                video_list.append(video_dir)

                for frame_idx in range(frames.shape[0]):
                    # BGR -> RGB
                    frame_rgb = cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)  # (H,W,3) uint8
                    img.save(os.path.join(video_dir, f"frame_{frame_idx:04d}.png"), format='PNG', quality=100)

            return video_list

        video_src_list = save_frames_pil(dataset_ori)
        video_wt_list = save_frames_pil(dataset_wt)
        result_videos = []
        for i in range (len(video_src_list)):
            result = self.evaluate_image(image_path=video_src_list[i],wt_image_path=video_wt_list[i],metric=metric)
            result_videos.append(result)
        return result_videos
        

    def _save_video(self, tensor, path):
        tensor = tensor.permute(0, 2, 3, 1).detach().cpu()
        tensor = (tensor * 255).to(torch.uint8)
        io.write_video(path, tensor.numpy(), fps=self.fps)


class VideoDataset(Dataset):
    def __init__(self, root_dir, crop_size, no_of_frames):
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.no_of_frames = no_of_frames
        self.video_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.mp4')]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Cannot read frames from: {video_path}")

        frame_indices = np.linspace(0, total_frames - 1, self.no_of_frames, dtype=int)
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            frame = self._crop_frame(frame, self.crop_size)
            if frame.size == 0:
                continue
            frames.append(frame)

        cap.release()

        if not frames:
            raise ValueError(f"No valid frames after cropping from: {video_path}")

        return np.stack(frames, axis=0)  # [T, H, W, 3], uint8


    def _crop_frame(self, frame, crop_size):
        # Several situations without cutting：None/False/0/(0,0)
        if not crop_size or (isinstance(crop_size, (tuple, list)) and (crop_size[0] == 0 or crop_size[1] == 0)):
            return frame

        h, w, _ = frame.shape
        if isinstance(crop_size, int):
            crop_h = crop_w = int(crop_size)
        else:
            crop_h, crop_w = map(int, crop_size)

        # Prevent boundary crossing: The cutting size cannot exceed the original size
        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)

        start_h = max((h - crop_h) // 2, 0)
        start_w = max((w - crop_w) // 2, 0)

        return frame[start_h:start_h + crop_h, start_w:start_w + crop_w, :]
