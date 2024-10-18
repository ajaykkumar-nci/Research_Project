import os
from pathlib import Path

def visdrone2yolo(dir):
    from PIL import Image
    from tqdm import tqdm

    def convert_box(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

    Path(os.path.join(dir, 'labels')).mkdir(parents=True, exist_ok=True)  # make labels directory
    pbar = tqdm(Path(os.path.join(dir, 'annotations')).glob('*.txt'), desc=f'Converting {dir}')
    for f in pbar:
        img_size = Image.open(Path(os.path.join(dir, 'images', f.name)).with_suffix('.jpg')).size
        lines = []
        with open(f, 'r') as file:
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if row[4] == '0':
                    continue
                cls = int(row[5]) - 1
                box = convert_box(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                with open(str(f).replace(f'{os.sep}annotations{os.sep}', f'{os.sep}labels{os.sep}'), 'w') as fl:
                    fl.writelines(lines)
    

def convert2yolo(data_path, d_list):
    for d in ['train','val', 'test']: 
        visdrone2yolo(os.path.join(data_path, d))

if __name__ == '__main__':
    convert2yolo("/home/ubuntu/Research_Project/Visdrone_images", ['train', 'val', 'test'])
    
