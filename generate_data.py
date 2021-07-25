import argparse

import torch
from torchvision import utils
from model_modified import Generator
from tqdm import tqdm
import random
import math
from PIL import Image
import numpy as np


def tensor2im(var):
    # var shape: (3, H, W)
    var = var.cpu().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))

def save_image(img, im_save_path):
    result = tensor2im(img)
    Image.fromarray(np.array(result)).save(im_save_path)

def generate(args, g_ema, device, mean_latent, size):
    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            
            # create 5 latent vectors
            list_sample = []
            for j in range(5):
                sample_z = torch.randn(1, args.latent, device=device)
                
                with torch.no_grad():
                    sample_w = g_ema.style(sample_z)
                """
                sample_w = g_ema.forward(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
                )
                """    
                
                #print(j,'th sample\n',sample_w[0], type(sample_w))
                #print(sample_w[0].size())
                list_sample.append(sample_w.unsqueeze(1).repeat(1,18,1))
                print('w: ',list_sample[-1].size())
                #print('\n',list_sample[j][0])

                '''
                # original image
                origin = g_ema.make_image(sample_w)
                utils.save_image(
                origin,
                f"sample/mixed_{str(i).zfill(6)}_{str(j)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
                )
                  '''

            # noise z 합치기
            # 붙이는 게 아니라 섞는거로 테스트해야 함.

            # select mixing index
            n_latent = (int(math.log(size,2)) * 2) - 2 #maybe 18

            injection = [0]
            n = 0
            for k in range(4):
                n = random.randint(n+1, 14+k)
                injection.append(n)

            #print('injection :',injection)
            
            # mixing
            latent = torch.cat([list_sample[0][0][0:injection[1]], list_sample[1][0][injection[1]:injection[2]]])
            latent = torch.cat([latent, list_sample[2][0][injection[2]:injection[3]]])
            latent = torch.cat([latent, list_sample[3][0][injection[3]:injection[4]]])
            latent = torch.cat([latent, list_sample[4][0][injection[4]:n_latent]])


            # mixed w latent -> save!
            latent = torch.unsqueeze(latent,0)
            #print('mixed latent size',latent.size())
            torch.save(latent, f"{args.output_dir}/latent/{str(i).zfill(6)}.pt")
            #torch.save(latent.state_dict(), f"latent/{str(i).zfill(6)}.pt")
            #print("latent is saved!")
            # make image
            #print(latent[:,0])
            """
            image = g_ema.make_image(latent)
            """
            print(latent.size())
            
            image = g_ema.make_image(latent,
                    noise=None,
                    randomize_noise=False)

            print('image size : ', image.size())
            # save image
            save_image(image[0], f"{args.output_dir}/sample/{str(i).zfill(6)}.png")
            

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="../pretrained_models/stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='stylerig_data',
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent, args.size)
