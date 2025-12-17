import numpy as np
import torch
import argparse
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
from attack.privacy.attacks.SD_MIA.dataset import load_pokemon_datasets, load_coco_datasets, load_flickr_datasets, load_ti_datasets
from sklearn import metrics
from transformers import BlipProcessor, BlipForConditionalGeneration
from attack.privacy.attacks.SD_MIA.filter import Fourier_filter


class Flag(object):
    pass


flags = Flag
unet = None
tokenizer = None
text_encoder = None
scheduler = None
vae = None


def init_model(diff_path, device):
    global unet, tokenizer, text_encoder, scheduler, vae

    vae = AutoencoderKL.from_pretrained(diff_path, subfolder='vae', use_auth_token=True)
    vae = vae.to(device)
    vae.eval()

    tokenizer = CLIPTokenizer.from_pretrained(diff_path, subfolder="tokenizer")
    scheduler = DDIMScheduler.from_pretrained(diff_path, subfolder="scheduler")

    text_encoder = CLIPTextModel.from_pretrained(diff_path, subfolder="text_encoder")
    text_encoder = text_encoder.to(device)

    unet = UNet2DConditionModel.from_pretrained(diff_path, subfolder='unet')
    unet = unet.to(device)
    unet.eval()


@torch.no_grad()
def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


@torch.no_grad()
def ddim_singlestep(model, FLAGS, x, t_c, t_target, requires_grad=False, device='cuda', encoder_hidden_states=None):
    # global CNT
    if encoder_hidden_states == None:
        print('Error! encoder_hidden_states==None')

    x = x.to(device)
    t_c = x.new_ones([x.shape[0], ], dtype=torch.long) * (t_c)
    t_target = x.new_ones([x.shape[0], ], dtype=torch.long) * (t_target)

    betas = scheduler.betas.double().to(device)
    alphas = 1. - betas
    alphas = torch.cumprod(alphas, dim=0)
    alphas_t_c = extract(alphas, t=t_c, x_shape=x.shape)
    alphas_t_target = extract(alphas, t=t_target, x_shape=x.shape)

    if requires_grad:
        epsilon = model(x, t_c, encoder_hidden_states).sample
    else:
        with torch.no_grad():
            epsilon = model(x, t_c, encoder_hidden_states).sample

    pred_x_0 = (x - ((1 - alphas_t_c).sqrt() * epsilon)) / (alphas_t_c.sqrt())
    x_t_target = alphas_t_target.sqrt() * pred_x_0 + (1 - alphas_t_target).sqrt() * epsilon

    return {'x_t_target': x_t_target, 'epsilon': epsilon}


@torch.no_grad()
def ddim_multistep(model, FLAGS, x, t_c, target_steps, clip=False, device='cuda', requires_grad=False,
                   encoder_hidden_states=None):
    for idx, t_target in enumerate(target_steps):
        result = ddim_singlestep(model, FLAGS, x, t_c, t_target, requires_grad=requires_grad, device=device,
                                 encoder_hidden_states=encoder_hidden_states)
        x = result['x_t_target']
        t_c = t_target

    if clip:
        result['x_t_target'] = torch.clip(result['x_t_target'], -1, 1)

    return result


@torch.no_grad()
def att_measure(diffusion, sample, metric, device='cuda'):
    diffusion = diffusion.to(device).float()
    sample = sample.to(device).float()

    if metric == 'l2':
        score = ((diffusion - sample) ** 2).flatten(1).sum(dim=-1)
    elif isinstance(metric, int):
        score = (torch.abs(diffusion - sample) ** metric).flatten(1).sum(dim=-1)
    else:
        raise NotImplementedError
    return score


@torch.no_grad()
def sec_mi(model, batch, vae, text_encoder, device, x_sec_list, x_sec_recon_list, Filter, threshold, scale, scores, highs, new_model, x_new_list, x_new_recon_list):
    target_steps = list(range(0, flags.t_sec + flags.timestep, flags.timestep))[1:]  # 10 20 ... 90 100
    starttmp = flags.t_sec

    batch["pixel_values"] = batch["pixel_values"].to(device)
    latents = vae.encode(batch["pixel_values"].to(torch.float32)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    x = latents
    embd_cond = text_encoder(batch["input_ids"].to(device))[0]

    x_sec = ddim_multistep(model, flags, x, t_c=0, target_steps=target_steps, device=device,
                           encoder_hidden_states=embd_cond)
    x_sec = x_sec['x_t_target']

    endtmp = starttmp + flags.stpsnumi

    forw_stps = list(range(starttmp, endtmp + 1))
    back_stps = list(reversed(list(range(starttmp, endtmp + 1))))

    embd = text_encoder(batch["input_ids"].to(device))[0]

    assert forw_stps[0] == 100 and forw_stps[1] == 101

    x_sec_forw = ddim_singlestep(model, flags, x_sec,
                                 t_c=forw_stps[0], t_target=forw_stps[1],
                                 device=device, encoder_hidden_states=embd)

    x_sec_recon = ddim_singlestep(model, flags, x_sec_forw['x_t_target'],
                                  t_c=back_stps[0], t_target=back_stps[1],
                                  device=device, encoder_hidden_states=embd)
    x_sec_recon = x_sec_recon['x_t_target']

    if Filter == 1:
        x_sec = Fourier_filter(x_sec, threshold=threshold, scale=scale)
        x_sec_recon = Fourier_filter(x_sec_recon, threshold=threshold, scale=scale)

    x_sec_list.append(x_sec)
    x_sec_recon_list.append(x_sec_recon)


@torch.no_grad()
def prox_mi(model, batch, vae, text_encoder, device, x_sec_list, x_sec_recon_list, Filter, threshold, scale):

    batch["pixel_values"] = batch["pixel_values"].to(device)
    latents = vae.encode(batch["pixel_values"].to(torch.float32)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    x = latents

    embd_cond = text_encoder(batch["input_ids"].to(device))[0]

    def prox_loss(model, flags, x, device, encoder_hidden_states, t=500):
        x = x.to(device)
        t = x.new_ones([x.shape[0], ], dtype=torch.long) * (t)
        betas = scheduler.betas.double().to(device)
        alphas = 1. - betas  ### α
        alphas = torch.cumprod(alphas, dim=0)  ### α
        alphas_t = extract(alphas, t=t, x_shape=x.shape)  ### α

        eps = model(x, 0, encoder_hidden_states).sample

        eps_pred = model(alphas_t.sqrt() * x + (1 - alphas_t).sqrt() * eps, t, encoder_hidden_states).sample

        eps_pred = (alphas_t.sqrt() * x + (1 - alphas_t).sqrt() * eps - (1 - alphas_t).sqrt() * eps_pred) / alphas_t.sqrt()
        eps = x
        return eps, eps_pred

    eps, eps_pred = prox_loss(model, flags, x, device, embd_cond, t=500)

    if Filter == 1:
        eps = Fourier_filter(eps, threshold=threshold, scale=scale)
        eps_pred = Fourier_filter(eps_pred, threshold=threshold, scale=scale)

    x_sec_list.append(eps)
    x_sec_recon_list.append(eps_pred)


@torch.no_grad()
def prox_mi_n(model, batch, vae, text_encoder, device, x_sec_list, x_sec_recon_list, Filter, threshold, scale):

    batch["pixel_values"] = batch["pixel_values"].to(device)
    latents = vae.encode(batch["pixel_values"].to(torch.float32)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    x = latents

    embd_cond = text_encoder(batch["input_ids"].to(device))[0]

    def prox_loss(model, flags, x, device, encoder_hidden_states, t=500):
        x = x.to(device)
        t = x.new_ones([x.shape[0], ], dtype=torch.long) * (t)
        betas = scheduler.betas.double().to(device)
        alphas = 1. - betas  ### α
        alphas = torch.cumprod(alphas, dim=0)  ### α
        alphas_t = extract(alphas, t=t, x_shape=x.shape)  ### α

        eps = model(x, 0, encoder_hidden_states).sample
        eps = eps / eps.abs().mean(list(range(1, eps.ndim)), keepdim=True) * (2 / torch.pi) ** 0.5

        eps_pred = model(alphas_t.sqrt() * x + (1 - alphas_t).sqrt() * eps, t, encoder_hidden_states).sample

        eps_pred = (alphas_t.sqrt() * x + (1 - alphas_t).sqrt() * eps - (1 - alphas_t).sqrt() * eps_pred) / alphas_t.sqrt()
        eps = x
        return eps, eps_pred

    eps, eps_pred = prox_loss(model, flags, x, device, embd_cond, t=500)

    if Filter == 1:
        eps = Fourier_filter(eps, threshold=threshold, scale=scale)
        eps_pred = Fourier_filter(eps_pred, threshold=threshold, scale=scale)

    x_sec_list.append(eps)
    x_sec_recon_list.append(eps_pred)


@torch.no_grad()
def loss_mi(model, batch, vae, text_encoder, device, x_sec_list, x_sec_recon_list, Filter, threshold, scale):

    batch["pixel_values"] = batch["pixel_values"].to(device)
    latents = vae.encode(batch["pixel_values"].to(torch.float32)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    x = latents

    embd_cond = text_encoder(batch["input_ids"].to(device))[0]

    def loss(model, flags, x, device, encoder_hidden_states, t=500): # 500
        x = x.to(device)
        t = x.new_ones([x.shape[0], ], dtype=torch.long) * (t)
        betas = scheduler.betas.double().to(device)
        alphas = 1. - betas  ### α
        alphas = torch.cumprod(alphas, dim=0)  ### α
        alphas_t = extract(alphas, t=t, x_shape=x.shape)  ### α

        eps = torch.randn_like(x)
        eps_pred = model(alphas_t.sqrt() * x + (1 - alphas_t).sqrt() * eps, t, encoder_hidden_states).sample

        eps_pred = (alphas_t.sqrt() * x + (1 - alphas_t).sqrt() * eps - (1 - alphas_t).sqrt() * eps_pred) / alphas_t.sqrt()
        eps = x

        return eps, eps_pred

    eps, eps_pred = loss(model, flags, x, device, embd_cond, t=500)

    if Filter == 1:
        eps = Fourier_filter(eps, threshold=threshold, scale=scale)
        eps_pred = Fourier_filter(eps_pred, threshold=threshold, scale=scale)

    x_sec_list.append(eps)
    x_sec_recon_list.append(eps_pred)


def SD_Attack(args):
    flags.T = 1000
    flags.train_batch_size = 1
    flags.dataloader_num_workers = 0
    flags.resolution = 512
    flags.image_column = "image"
    flags.caption_column = "text"
    flags.t_sec = 100
    flags.timestep = 10
    flags.stpsnumi = 1
    flags.black = 1 # black-box setting

    flags.attack = args.method
    flags.dataset = args.dataset
    flags.filter = args.filter

    device = args.device
    t = 5
    s = 0.2

    assert flags.attack in ['sec', 'naive', 'pia', 'pian']
    assert flags.dataset in ['pokemon', 'coco', 'flickr', 'text-to-image-2m']

    flags.diff_path = args.model_path
    dataset_root = "/home/puwei_lian/storage/datasets/"

    init_model(flags.diff_path, device)
    print("loading finish!")

    if flags.dataset == 'pokemon':
        _, _, train_loader, test_loader = load_pokemon_datasets(dataset_root, num_samples=args.num_sample, tokenizer=tokenizer)
    elif flags.dataset == 'coco':
        _, _, train_loader, test_loader = load_coco_datasets(dataset_root, num_samples=args.num_sample, tokenizer=tokenizer)
    elif flags.dataset == 'flickr':
        train_loader, test_loader = load_flickr_datasets(dataset_root, num_samples=args.num_sample, tokenizer=tokenizer)
    elif flags.dataset == 'text-to-image-2m':
        train_loader, test_loader = load_ti_datasets(dataset_root, num_samples=args.num_sample, tokenizer=tokenizer)
    else:
        raise NotImplementedError

    x_sec_list = []
    x_sec_recon_list = []

    print(f"Start Attack! - Method: {flags.attack}, Dataset: {flags.dataset}")

    for step, batch in enumerate(tqdm(train_loader)):
        if flags.attack == 'naive':
            loss_mi(unet, batch, vae, text_encoder, device, x_sec_list, x_sec_recon_list, flags.filter, threshold=t, scale=s)
            flags.black = 0
        elif flags.attack == 'sec':
            sec_mi(unet, batch, vae, text_encoder, device, x_sec_list, x_sec_recon_list, flags.filter, threshold=t, scale=s)
            flags.black = 0
        elif flags.attack == 'pia':
            prox_mi(unet, batch, vae, text_encoder, device, x_sec_list, x_sec_recon_list, flags.filter, threshold=t, scale=s)
            flags.black = 0
        elif flags.attack == 'pian':
            prox_mi_n(unet, batch, vae, text_encoder, device, x_sec_list, x_sec_recon_list, flags.filter, threshold=t, scale=s)
            flags.black = 0
        else:
            print('Error, No implement!', flags.attack)
            exit()

    x_sec_s = torch.concat(x_sec_list)
    x_sec_recon_s = torch.concat(x_sec_recon_list)
    norm = 'l2'
    member_scores = att_measure(x_sec_s, x_sec_recon_s, norm, device=device).cpu()

    print("******************************************")
    x_sec_list = []
    x_sec_recon_list = []

    for step, batch in enumerate(tqdm(test_loader)):
        if flags.attack == 'sec':
            sec_mi(unet, batch, vae, text_encoder, device, x_sec_list, x_sec_recon_list, flags.filter, threshold=t, scale=s)
        elif flags.attack == 'pia':
            prox_mi(unet, batch, vae, text_encoder, device, x_sec_list, x_sec_recon_list, flags.filter, threshold=t, scale=s)
        elif flags.attack == 'naive':
            loss_mi(unet, batch, vae, text_encoder, device, x_sec_list, x_sec_recon_list, flags.filter, threshold=t, scale=s)
        elif flags.attack == 'pian':
            prox_mi_n(unet, batch, vae, text_encoder, device, x_sec_list, x_sec_recon_list, flags.filter, threshold=t, scale=s)
        else:
            print('Error, No implement!', flags.attack)
            exit()

    x_sec_s = torch.concat(x_sec_list)
    x_sec_recon_s = torch.concat(x_sec_recon_list)

    norm = 'l2'
    nonmember_scores = att_measure(x_sec_s, x_sec_recon_s, norm, device=device).cpu()

    min_score = min(member_scores.min(), nonmember_scores.min())
    max_score = max(member_scores.max(), nonmember_scores.max())

    TPR_list = []
    FPR_list = []

    TPRatFPR_1 = 0
    FPR_1_idx = 999
    TPRatFPR_01 = 0
    FPR_01_idx = 999

    total = member_scores.size(0) + nonmember_scores.size(0)
    max_acc = 0.0
    best_threshold = 0.0

    for threshold in torch.range(min_score, max_score, (max_score - min_score) / 10000):
        acc = ((member_scores <= threshold).sum() + (nonmember_scores > threshold).sum()) / total

        TP = (member_scores <= threshold).sum()
        TN = (nonmember_scores > threshold).sum()
        FP = (nonmember_scores <= threshold).sum()
        FN = (member_scores > threshold).sum()

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        if FPR_1_idx > (0.01 - FPR).abs():
            FPR_1_idx = (0.01 - FPR).abs()
            TPRatFPR_1 = TPR

        if FPR_01_idx > (0.001 - FPR).abs():
            FPR_01_idx = (0.001 - FPR).abs()
            TPRatFPR_01 = TPR

        TPR_list.append(TPR.item())
        FPR_list.append(FPR.item())

        if acc > max_acc:
            max_acc = acc
            best_threshold = threshold.item()
        # print(f'Score threshold = {threshold:.16f} \t ASR: {acc:.8f} \t TPR: {TPR:.8f} \t FPR: {FPR:.8f}')

    auc = metrics.auc(np.asarray(FPR_list), np.asarray(TPR_list))
    print(f'AUC: {auc} \t ASR: {max_acc} \t TPR@FPR=1%: {TPRatFPR_1} \t TPR@FPR=0.1%: {TPRatFPR_01}')
    print(f'Threshold:{best_threshold}')

    return auc, FPR_list, TPR_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--method', default='black_score_t', type=str, choices=['naive', 'pia', 'pian', 'sec', 'black_naive', 'black_score_t', 'black_score_c', 'black_score_d'])
    parser.add_argument('--num_sample', default=10, type=int)
    parser.add_argument('--dataset', default='coco', type=str, choices=['coco', 'pokemon', 'flickr', 'text-to-image-2m'])
    parser.add_argument('--model_path', default='/home/hub/model/stable-diffusion-v1-4', type=str, help='model path')
    parser.add_argument('--filter', default=0, type=int, help='Improvements in naive/sec/pia/pian')  # https://arxiv.org/abs/2505.20955
    parser.add_argument('--device', default='cuda:0', type=str)

    args = parser.parse_args()
    SD_Attack(args)






