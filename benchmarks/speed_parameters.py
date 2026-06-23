import os
import sys
import argparse
sys.path.append('.')
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format
from utils import read, rgb_to_yuv420

parser = argparse.ArgumentParser(description='IFRNet Speed and Parameters Benchmark')
parser.add_argument('--model_name', default='IFRNet', type=str, help='IFRNet, IFRNet_L, IFRNet_S, IFRNet_RGB2')
args = parser.parse_args()

yuv_input = False

WIDTH = 1920
HEIGHT = 1088

if args.model_name == 'IFRNet':
    from models.IFRNet import Model
elif args.model_name == 'IFRNet_L':
    from models.IFRNet_L import Model
elif args.model_name == 'IFRNet_S':
    from models.IFRNet_S import Model
elif args.model_name == 'IFRNet_RGB2':
    from models.IFRNET_RGB2 import Model
elif args.model_name == 'IFRNet_RGB2P':
    from models.IFRNET_RGB2P import Model
elif args.model_name == 'IFRNet_YUV420':
    from models.IFRNet_YUV420 import Model
    yuv_input = True
elif args.model_name == 'IFRNet_YUV420P':
    from models.IFRNet_YUV420P import Model
    yuv_input = True
elif args.model_name == 'IFRNet_YUV420F':
    from models.IFRNet_YUV420 import Model
    yuv_input = True
elif args.model_name == 'IFRNet_YUV420FP':
    from models.IFRNet_YUV420P import Model
    yuv_input = True
else:
    print(f"Unknown model: {args.model_name}")
    quit(-1)


if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

img0 = torch.from_numpy(read('figures/img0.png')).permute(2, 0, 1).unsqueeze(0).cuda() / 255.0
img1 = torch.from_numpy(read('figures/img1.png')).permute(2, 0, 1).unsqueeze(0).cuda() / 255.0
img0 = torch.nn.functional.interpolate(img0, (HEIGHT, WIDTH), mode='bilinear')
img1 = torch.nn.functional.interpolate(img1, (HEIGHT, WIDTH), mode='bilinear')
embt = torch.tensor(1/2).float().view(1, 1, 1, 1).cuda()

img0_y, img0_uv = rgb_to_yuv420(img0)
img1_y, img1_uv = rgb_to_yuv420(img1)

model = Model().cuda().eval()
original_forward = model.forward

print('Calculating FLOPs and Params...')
model.forward = model.inference_yuv if yuv_input else model.inference
inputs = (img0_y, img0_uv, img1_y, img1_uv, embt) if yuv_input else (img0, img1, embt)
height, width = img0.shape[2:4]
flops, params = profile(model, inputs=inputs)
flops, params = clever_format([flops, params], "%.3f")
print(f'FLOPs: {flops}, Params (by thop): {params}, for image: {width}x{height}')

total = sum([param.nelement() for param in model.parameters()])
print('Parameters (manual count): {:.2f}M'.format(total / 1e6))


NUM_ITERATIONS = 1000

with torch.inference_mode():
    # inference = model.inference_yuv if yuv_input else model.inference
    traced_model = torch.jit.optimize_for_inference(torch.jit.trace(model, inputs))
    # Прогрев
    for i in range(10):
        out = traced_model(*inputs) 
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_stamp = time.perf_counter()
    
    # Чистый замер
    for i in range(NUM_ITERATIONS):
        out = traced_model(*inputs) 
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print('Inference Time: {:.2f}ms / frame'.format(1000.0 * (time.perf_counter() - time_stamp) / NUM_ITERATIONS))

model.forward = original_forward
