import cv2
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
import numpy as np
import os
import argparse
from strhub.models.utils import load_from_checkpoint, parse_model_args
from pathlib import Path
from strhub.data.module import SceneTextDataModule
from PIL import Image
from tqdm import tqdm
import yaml
from unidecode import unidecode
def Processing(img,cuda=True,w=''):
    #prepare Reg Model:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cased', action='store_true', default=False, help='Cased comparison')
    parser.add_argument('--punctuation', action='store_true', default=False, help='Check punctuation')
    parser.add_argument('--new', action='store_true', default=False, help='Evaluate on new benchmark datasets')
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--custom', action='store_true', default=True, help='Evaluate on custom dataset')
    parser.add_argument('--img_dir', type=str, required=False, help='Path to image folder')
    parser.add_argument('--txt_dir', type=str, required=False, help='Path to gt folder')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    args.device = config['device']
    args.img_dir = config['img_dir']
    args.txt_dir = config['txt_dir']
    args.checkpoint = config['checkpoint']
    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size, augment=False)


    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    session = ort.InferenceSession(w, providers=providers)
    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    names = ['text']
    # colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image = img.copy()
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255
    im.shape

    outname = [i.name for i in session.get_outputs()]
    outname

    inname = [i.name for i in session.get_inputs()]
    inname

    inp = {inname[0]:im}
    # ONNX inference
    outputs = session.run(outname, inp)[0]
    ori_images = [img.copy()]

    for _,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
        image = ori_images[int(batch_id)]
        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score),3)

        count = 0
        crop_im = image.crop((box[0], box[1], box[2], box[3]))
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
        cv2.rectangle(image,box[:2],box[2:],[225, 255, 255],2)
        cv2.putText(image,pred_text,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)  
    return Image.fromarray(ori_images[0])
    
    