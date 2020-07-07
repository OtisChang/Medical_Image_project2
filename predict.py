import argparse
import logging
import os

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2 

from unet import UNet
from FCN.segmentation import fcn_resnet101, deeplabv3_resnet50

from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset

# modelPath = './checkpoints/CP_epoch50.pth' # Your model's path
modelPath = './checkpoints_fcn_noPretrain/5_150e_batch1_valid45/CP_epoch150.pth'


def predict_img(net,
                full_img,
                device,
                file_name,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)['out']

        if net.num_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)
        save_array = probs.cpu().numpy()
        mat_prob=np.reshape(save_array,[300,300])
        save_fn = 'D:/users/otis/MedicalImage_Project02_Segmentation/MedicalImage_Project02_Segmentation/private_data_10/private_data_10/Results' + file_name[:-4] + '_prob.mat'
        sio.savemat(save_fn, {'array' : mat_prob})
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default=modelPath,
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', default='0111_01_3.png', metavar='INPUT', nargs='+',
                        help='filenames of input images')

    parser.add_argument('--output', '-o', default='0111_01_3_out.png', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=True)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.25)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    print("Type of in_files = ", type(in_files))
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    dirPath = "D:/users/otis/MedicalImage_Project02_Segmentation/MedicalImage_Project02_Segmentation/private_data_10/private_data_10/imgs2"
    img_name = os.listdir(dirPath)
    in_path = []
    out_fileName = []
    for i in range(len(img_name)):
        print(img_name[i])
        path = dirPath + "/" + img_name[i]
        output_name = "out_" + img_name[i]
        out_fileName.append(output_name)
        in_path.append(path)

    args = get_args()
    # in_files = args.input
    # out_files = get_output_filenames(args)
    in_files = in_path
    out_files = out_fileName

    # net = UNet(n_channels=3, n_classes=1)
    net = fcn_resnet101(num_classes=1)
    # net = deeplabv3_resnet50(num_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))
        img = Image.open(fn)
        img=img.convert(mode='RGB')

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device,
                           file_name=out_files[i])

        if not args.no_save:
            out_fn = out_files[i]
            print("out_files = ", out_files[i])
            result = mask_to_image(mask)
            result.save("D:/users/otis/MedicalImage_Project02_Segmentation/MedicalImage_Project02_Segmentation/private_data_10/private_data_10/Results" + out_files[i])

            logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            # plot_img_and_mask(img, mask)
