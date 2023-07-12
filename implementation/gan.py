import os 
import sys 

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import argparse 
from trainer.gan import Trainer_GAN 


parser = argparse.ArgumentParser()
parser.add_argument("--path_image", type=str, default='D:/Datasets/CelebA/img_align_celeba/', help="path of dataset")
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--dim_latent", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--channel_g", type=int, default=64, help="number of generator channels")
parser.add_argument("--channel_d", type=int, default=64, help="number of discriminator channels")
parser.add_argument("--step_g", type=int, default=1, help="step of generator")
parser.add_argument("--load_model", type=bool, default=False, help="step of generator")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--path_result", type=str, default='.result/', help="path of result")


if __name__ == '__main__':
    opt = parser.parse_args()
    print(opt)
    gan = Trainer_GAN(opt)
    gan.train()
