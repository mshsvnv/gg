"""Интерполяция промежуточного кадра между 0 и 2 каждой моделью IFRNet.

Для каждой сцены в examples/ синтезируется кадр на середине между 0 и 2
(соответствует ground-truth кадру 1) каждой из четырёх моделей, результат
сохраняется как 1_<model>.png. Дополнительно собираются gif-последовательности
и монтажи визуального сравнения с эталоном для отчёта.
"""
import os
import os.path as osp
import glob
import importlib
import cv2
import numpy as np
import torch
import imageio.v2 as imageio

DEVICE = torch.device('cpu')

# имя модели -> (python-модуль, путь к чекпоинту, подпись для монтажа)
MODELS = {
    'IFRNet':        ('models.IFRNet',        './checkpoint/IFRNet/IFRNet_best.pth',             'IFRNet'),
    'IFRNet_RGB2':   ('models.IFRNET_RGB2',   './checkpoint/IFRNet_RGB2/IFRNet_RGB2_best.pth',   'RGB2'),
    'IFRNet_YUV420': ('models.IFRNet_YUV420', './checkpoint/IFRNet_YUV420/IFRNet_YUV420_best.pth','YUV420'),
    'IFRNet_RGB2P':  ('models.IFRNET_RGB2P',  './checkpoint/IFRNet_RGB2P/IFRNet_RGB2P_best.pth', 'RGB2+LPIPS'),
}

EXAMPLES_DIR = './examples'
GIF_DIR = './examples/gif_results'
REPORT_IMG = './nir_report/inc/img'


def read_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_tensor(np_img):
    return (torch.tensor(np_img.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(DEVICE)


def pad_to(t, mult=16):
    _, _, h, w = t.shape
    ph = (mult - h % mult) % mult
    pw = (mult - w % mult) % mult
    return torch.nn.functional.pad(t, (0, pw, 0, ph), mode='replicate'), h, w


def load_model(module_name, ckpt_path):
    Model = importlib.import_module(module_name).Model
    model = Model().to(DEVICE).eval()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f'  [warn] missing={len(missing)} unexpected={len(unexpected)} keys')
    return model


def interpolate(model, img0_np, img2_np):
    img0, img2 = to_tensor(img0_np), to_tensor(img2_np)
    img0p, h, w = pad_to(img0)
    img2p, _, _ = pad_to(img2)
    embt = torch.tensor(0.5).view(1, 1, 1, 1).float().to(DEVICE)
    with torch.no_grad():
        pred = model.inference(img0p, img2p, embt)
    pred = pred[:, :, :h, :w]
    return (pred[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)


def find_frame(folder, idx):
    for p in sorted(glob.glob(osp.join(folder, f'{idx}.*'))):
        if osp.splitext(p)[1].lower() in ('.png', '.jpg', '.jpeg'):
            return p
    return None


def label_tile(img_bgr, text, tile_h=240, barh=32):
    h, w = img_bgr.shape[:2]
    im = cv2.resize(img_bgr, (int(w * tile_h / h), tile_h), interpolation=cv2.INTER_AREA)
    w = im.shape[1]
    bar = np.full((barh, w, 3), 255, np.uint8)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(bar, text, ((w - tw) // 2, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, im])


def montage(folder, out_path):
    """Ряд: GT (кадр 1) и интерполяции каждой модели."""
    tiles = [label_tile(cv2.imread(find_frame(folder, 1)), 'GT')]
    for name, (_, _, lab) in MODELS.items():
        tiles.append(label_tile(cv2.imread(osp.join(folder, f'1_{name}.png')), lab))
    pad = 6
    H = max(t.shape[0] for t in tiles)
    row = []
    for i, t in enumerate(tiles):
        row.append(t)
        if i < len(tiles) - 1:
            row.append(np.full((H, pad, 3), 255, np.uint8))
    cv2.imwrite(out_path, np.hstack(row))


def main():
    os.makedirs(GIF_DIR, exist_ok=True)
    os.makedirs(REPORT_IMG, exist_ok=True)

    folders = []
    for nm in sorted(os.listdir(EXAMPLES_DIR)):
        p = osp.join(EXAMPLES_DIR, nm)
        if osp.isdir(p) and nm != 'gif_results' and find_frame(p, 0) and find_frame(p, 2):
            folders.append((nm, p))

    for model_name, (module, ckpt, _) in MODELS.items():
        print(f'\n=== {model_name} ===')
        model = load_model(module, ckpt)
        for nm, p in folders:
            f0, f2 = find_frame(p, 0), find_frame(p, 2)
            img0, img2 = read_rgb(f0), read_rgb(f2)
            mid = interpolate(model, img0, img2)
            out = osp.join(p, f'1_{model_name}.png')
            cv2.imwrite(out, cv2.cvtColor(mid, cv2.COLOR_RGB2BGR))
            gif = osp.join(GIF_DIR, f'interp_{model_name}_{nm}.gif')
            imageio.mimsave(gif, [img0, mid, img2], fps=3, loop=0)
            print(f'  {nm}: {osp.basename(out)} -> {osp.basename(gif)}')

    # монтажи визуального сравнения для отчёта
    montage(osp.join(EXAMPLES_DIR, 'car'),  osp.join(REPORT_IMG, 'visual_comparison1.png'))
    montage(osp.join(EXAMPLES_DIR, 'hair'), osp.join(REPORT_IMG, 'visual_comparison2.png'))
    print('\nМонтажи сохранены в', REPORT_IMG)


if __name__ == '__main__':
    main()
