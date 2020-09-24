import numpy as np
from PIL import Image, ImageOps


def letterbox_image(image: np.ndarray, image_h: int, image_w: int, fill_color: int = 0) -> np.ndarray:
    if isinstance(image, np.ndarray):
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
    tgt_w, tgt_h = image_w, image_h
    src_w, src_h = image.size

    if src_w == tgt_w and src_h == tgt_h:
        return np.array(image)

    if src_w / tgt_w >= src_h / tgt_h:
        scale = tgt_w / src_w
    else:
        scale = tgt_h / src_h
    if scale != 1:
        bands = image.split()
        bands = [b.resize((int(scale * src_w), int(scale * src_h)), resample=Image.LANCZOS) for b in bands]
        image = Image.merge(image.mode, bands)
        src_w, src_h = image.size

    if src_w == tgt_w and src_h == tgt_h:
        return np.array(image)

    # padding
    img_np = np.array(image)
    channels = img_np.shape[2] if len(img_np.shape) > 2 else 1
    pad_w = (tgt_w - src_w) / 2
    pad_h = (tgt_h - src_h) / 2
    pad = (int(pad_w), int(pad_h), int(pad_w + 0.5), int(pad_h + 0.5))
    image = ImageOps.expand(image, border=pad, fill=(fill_color,) * channels)
    return np.array(image)
