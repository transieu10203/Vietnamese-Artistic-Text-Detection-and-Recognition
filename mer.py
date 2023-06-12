import os
import torch
from torchvision.ops import nms
TXT_PATH = "/content/UAIC_2022_Kronus/Detection/mer" 
PATH = os.listdir(TXT_PATH)
SAVE_PATH = '/content/UAIC_2022_Kronus/Submit' 
IMAGE_PATH = '/content/data/uaic2022_private_test/images'
IOU_THRES = 0.3
os.makedirs(SAVE_PATH,exist_ok=True)
for image in sorted(os.listdir(IMAGE_PATH)):
    txt_name = image.replace('jpg','txt')

    bbox = []
    score = []

    for  _path in sorted(PATH):
        path = os.path.join(TXT_PATH,_path)
        folder_path = os.path.join(path, txt_name)
        if not os.path.isfile(folder_path):
            continue
        with open(folder_path,encoding='utf-8') as f:
            data = f.readlines()
        data = [i.strip().split(',') for i in data]
        for line in data:
            try:
                converted_line = list(map(float,line))
            except:
                converted_line = list(map(float,line[:-1]))
                converted_line.append(float(line[-1].replace('tensor(','').replace(", device='cuda:0')","")))
            b = converted_line[:-1]
            bbox.append([b[0],b[5],b[2],b[1]])
            score.append(converted_line[-1])
    if len(bbox) == 0:
        with open(os.path.join(SAVE_PATH,txt_name),'w') as f:
            f.write("")
        continue
    
    t_bbox = torch.FloatTensor(bbox)
    score = torch.FloatTensor(score)
    pred = nms(t_bbox,score, IOU_THRES)
    new_bboxes = t_bbox[pred.tolist()].tolist()
    new_bboxes = [ list(map(int,i)) for i in new_bboxes ]
    
    res = ""
    for b in new_bboxes:
        coors = [str(b[0]),str(b[3]),str(b[2]),str(b[3]),str(b[2]),str(b[1]),str(b[0]),str(b[1])]
        res += ','.join(coors)
        res += ',\n'
    
    with open(os.path.join(SAVE_PATH,txt_name),'w') as f:
        f.write(res)



    
