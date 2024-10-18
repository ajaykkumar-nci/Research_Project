import os
from pathlib import Path
import shutil
import pandas as pd
import numpy as np

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


def video2images(IpAnnPath, IpImgPath, OpAnnPath, OpImgPath):

    def arrageImgs(IpImgPath, OpImgPath):
        Path(OpImgPath).mkdir(parents=True, exist_ok=True)
        image_sequences = os.listdir(IpImgPath)
        for seq in image_sequences:
            for img in os.listdir(os.path.join(IpImgPath, seq)):
                shutil.move(os.path.join(IpImgPath, seq, img), os.path.join(OpImgPath, seq+"_"+img))

    def arrageAnns(IpAnnPath, OpAnnPath):
        Path(OpAnnPath).mkdir(parents=True, exist_ok=True)
        image_annotations = os.listdir(IpAnnPath)
        padding_length = 7 # got this from the image name
        for ann in image_annotations:
            df = pd.read_csv(os.path.join(IpAnnPath,ann), header=None)
            unique_df = df[0].unique()
            for i in range(len(unique_df)):
                # naming the txt file
                txt_name = str(unique_df[i]).zfill(padding_length) + ".txt"
                ann_name = ann.split(".")[0]+"_"+txt_name

                # get annotations for each image
                filtered_df = df[df[0] == 1]
                with open(os.path.join(OpAnnPath, ann_name), 'w') as f:
                    for _, row in filtered_df.iterrows():
                        ann_lines = ",".join([str(i) for i in row[2:]])
                        f.write(ann_lines + '\n')

    arrageImgs(IpImgPath, OpImgPath)
    arrageAnns(IpAnnPath, OpAnnPath)
    

def convert2yolo(data_path, d_list):
    for d in ['train','val', 'test']: 
        visdrone2yolo(os.path.join(data_path, d))

def processVisDroneVID(data_path, d_list):
    for d in d_list: 
        video2images(os.path.join(data_path, d, 'old_annotations'),
                     os.path.join(data_path, d, 'sequences'),
                     os.path.join(data_path, d, 'annotations'),
                     os.path.join(data_path, d, 'images'))

def combineDatasets(video_path, image_path, output_path):
    def moveFiles(data_list, input_path, output_path):
        for d in data_list:
            if d.split('-')[-1] == 'train':
                for img in os.listdir(os.path.join(input_path, d, 'images')):
                    shutil.move(os.path.join(input_path, d, 'images', img), os.path.join(output_path, 'train', 'images',  img))
                for ann in os.listdir(os.path.join(input_path, d, 'labels')):
                    shutil.move(os.path.join(input_path, d, 'labels', ann), os.path.join(output_path, 'train', 'labels', ann)) 
            elif d.split('-')[-1] == 'val':
                for img in os.listdir(os.path.join(input_path, d, 'images')):
                    shutil.move(os.path.join(input_path, d, 'images', img), os.path.join(output_path, 'val', 'images', img))
                for ann in os.listdir(os.path.join(input_path, d, 'labels')):
                    shutil.move(os.path.join(input_path, d, 'labels', ann), os.path.join(output_path, 'val', 'labels', ann))
            else:
                for img in os.listdir(os.path.join(input_path, d, 'images')):
                    shutil.move(os.path.join(input_path, d, 'images', img), os.path.join(output_path, 'test', 'images', img))
                for ann in os.listdir(os.path.join(input_path, d, 'labels')):
                    shutil.move(os.path.join(input_path, d, 'labels', ann), os.path.join(output_path, 'test', 'labels', ann))
    Path(os.path.join(output_path, 'train', 'images')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_path, 'train', 'labels')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_path, 'val', 'images')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_path, 'val', 'labels')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_path, 'test', 'images')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_path, 'test', 'labels')).mkdir(parents=True, exist_ok=True)

    moveFiles(['train','val', 'test'], video_path, output_path)
    moveFiles(['train', 'val', 'test'], image_path, output_path)

if __name__ == '__main__':
    processVisDroneVID("/home/ubuntu/Research_Project/yolo_models/Visdrone_video", ['train','val', 'test'])
    convert2yolo("/home/ubuntu/Research_Project/yolo_models/Visdrone_video", ['train','val', 'test'])
    combineDatasets("/home/ubuntu/Research_Project/yolo_models/Visdrone_video", "/home/ubuntu/Research_Project/yolo_models/Visdrone", "/home/ubuntu/Research_Project/yolo_models/VisCombined")
