import argparse
from strhub.models.utils import load_from_checkpoint, parse_model_args
from pathlib import Path
from strhub.data.module import SceneTextDataModule
from PIL import Image
from tqdm import tqdm
from unidecode import unidecode
import os
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
parser.add_argument('--checkpointsub', help="Model checkpoint (or 'pretrained=<model_id>')")
parser.add_argument('--data_root', default='data')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--cased', action='store_true', default=False, help='Cased comparison')
parser.add_argument('--punctuation', action='store_true', default=False, help='Check punctuation')
parser.add_argument('--new', action='store_true', default=False, help='Evaluate on new benchmark datasets')
parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
parser.add_argument('--device', default='cpu')
parser.add_argument('--custom', action='store_true', default=True, help='Evaluate on custom dataset')
parser.add_argument('--img_dir', type=str, required=True, help='Path to image folder')
parser.add_argument('--txt_dir', type=str, required=True, help='Path to gt folder')
args, unknown = parser.parse_known_args()
kwargs = parse_model_args(unknown)
model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
img_transform = SceneTextDataModule.get_transform(model.hparams.img_size, augment=False)
count = 0
for img_f in tqdm(os.listdir(args.img_dir)):
    with open(os.path.join(args.txt_dir,img_f.replace('.jpg','.txt')),'r',encoding='utf-8') as f:
        data = f.readlines()
    data = [i.strip() for i in data]
    res = ""
    img = Image.open(os.path.join(args.img_dir,img_f)).convert('RGB')
    for line in data:
        bbox = list(map(int,line.split(',')[:-1]))
        crop_im = img.crop((bbox[0], bbox[5], bbox[2], bbox[1]))
        # crop_im.save("test.jpg")
        crop_im = img_transform(crop_im).unsqueeze(0)
        crop_im = crop_im.to(args.device)
        logits = model(crop_im)
        pred = logits.softmax(-1)
        pred_strs, confidence = model.tokenizer.decode(pred)
        num_fail = 0
        for conf in confidence[0].tolist():
            if conf < 0.5:
                num_fail += 1
        
        if num_fail >= (len(confidence[0]) // 2):
            count += 1
            continue

        pred_text = pred_strs[0]
        # pred_text = unidecode(pred_text)
        pred_text=str(pred_text)
        res += ",".join(line.split(',')[:-1])
        res +=","+pred_text +'\n'
    with open(os.path.join(args.txt_dir,img_f.replace('.jpg','.txt')),'w',encoding='utf-8') as f:
        f.write(res)
    # except:
    #     with open(os.path.join(args.txt_dir,img_f.replace('.jpg','.txt')),'w',encoding='utf-8') as f:
    #         f.write("")
print(count)

