import os
from imageio import mimsave
import imageio.v2 as imageio  # Явно указываем версию API

for dir_name in os.listdir('.'):
    # Пропускаем файлы, а не директории
    if not os.path.isdir(dir_name):
        continue
    
    # Проверяем существование файлов
    img_00_path = f'./{dir_name}/0.jpg'
    img_01_path = f'./{dir_name}/1.jpg'
    img_02_path = f'./{dir_name}/2.jpg'
    
    img_ifrnet_path = f'./{dir_name}/0_5_ifrnet.png'
    img_rife_path = f'./{dir_name}/0_5_rife.png'
    img_sepconv_path = f'./{dir_name}/0_5_sepconv.png'
    img_vfiformer_path = f'./{dir_name}/0_5_vfiformer.png'
    
    print(f"Обрабатываем: {dir_name}")
    
    try:
        # Загружаем изображения
        img_00 = imageio.imread(img_00_path)
        img_01 = imageio.imread(img_01_path)
        img_02 = imageio.imread(img_02_path)
        
        img_ifrnet = imageio.imread(img_ifrnet_path)
        img_rife = imageio.imread(img_rife_path)
        img_sepconv = imageio.imread(img_sepconv_path)
        img_vfiformer = imageio.imread(img_vfiformer_path)
        
        # Создаём директорию для GIF, если её нет
        gif_dir = './gif_results'
        os.makedirs(gif_dir, exist_ok=True)
        
        # 1. Ground Truth GIF (I0 → I1 → I2)
        gt_gif_path = f'{gif_dir}/gt_{dir_name}.gif'
        mimsave(gt_gif_path, [img_00, img_01, img_02], fps=3, loop=0)
        print(f"  Создан: {gt_gif_path}")
        
        # 2. IFRNet сравнение GIF (I0 → IFRNet → I2)
        ifrnet_gif_path = f'{gif_dir}/ifrnet_{dir_name}.gif'
        mimsave(ifrnet_gif_path, [img_00, img_ifrnet, img_02], fps=3, loop=0)
        print(f"  Создан: {ifrnet_gif_path}")
        
        # 3. RIFE сравнение GIF (I0 → RIFE → I2)
        rife_gif_path = f'{gif_dir}/rife_{dir_name}.gif'
        mimsave(rife_gif_path, [img_00, img_rife, img_02], fps=3, loop=0)
        print(f"  Создан: {rife_gif_path}")
        
        # 4. SepConv сравнение GIF (I0 → SepConv → I2)
        sepconv_gif_path = f'{gif_dir}/sepconv_{dir_name}.gif'
        mimsave(sepconv_gif_path, [img_00, img_sepconv, img_02], fps=3, loop=0)
        print(f"  Создан: {sepconv_gif_path}")
        
        # 5. VFIformer сравнение GIF (I0 → VFIformer → I2)
        vfiformer_gif_path = f'{gif_dir}/vfiformer_{dir_name}.gif'
        mimsave(vfiformer_gif_path, [img_00, img_vfiformer, img_02], fps=3, loop=0)
        print(f"  Создан: {vfiformer_gif_path}")
        
        
    except Exception as e:
        print(f"Ошибка при обработке {dir_name}: {e}")
        continue

print("Обработка завершена!")