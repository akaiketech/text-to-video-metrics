import numpy as np
import torch
from tqdm import tqdm

def trans(x):
    # If grayscale images, add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # Permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4) 

    return x

def calculate_fvd(videos1, videos2, device, method='styleganv', only_final=False):

    if method == 'styleganv':
        from fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
    elif method == 'videogpt':
        from fvd.videogpt.fvd import load_i3d_pretrained, frechet_distance
        from fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats

    print("calculate_fvd...")

    assert videos1.shape == videos2.shape, "Both videos must have the same shape"

    i3d = load_i3d_pretrained(device=device)

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    fvd_results = []

    if only_final:
        assert videos1.shape[2] >= 10, "For calculating FVD, each clip must have at least 10 frames."

        feats1 = get_fvd_feats(videos1, i3d=i3d, device=device)
        feats2 = get_fvd_feats(videos2, i3d=i3d, device=device)

        fvd_results.append(frechet_distance(feats1, feats2))
    
    else:
        for clip_timestamp in tqdm(range(10, videos1.shape[-3] + 1)):
            videos_clip1 = videos1[:, :, :clip_timestamp]
            videos_clip2 = videos2[:, :, :clip_timestamp]

            feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
            feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)

            fvd_results.append(frechet_distance(feats1, feats2))

    # Compute the average FVD score
    average_fvd = np.mean(fvd_results)

    return {"average_fvd": average_fvd}


# test code / using example

