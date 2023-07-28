import torch


def blend(a, b, s, look_back, overlap):
    """
    Blend the
    Parameters
    ----------
    0              s - look_back                s
    |              |                            |
        motion a
                     interp. between a and b
                                                    b
    Returns
    -------
    """
    if s - look_back <= 0:
        return b

    b = torch.cat([a[..., :-overlap], b], dim=-1)
    # res = torch.empty(a.shape[:-1] + (a.shape[-1] + b.shape[-1] - overlap), device=a.device)
    res = b.clone()
    weight = torch.linspace(1, 0, look_back).to(a.device)
    weight = weight[None][None][None]
    res[..., s-look_back:s] = a[..., s-look_back:s] * weight + b[..., s-look_back:s] * (1 - weight)
    res[..., :s-look_back] = a[..., :s-look_back]
    return res
