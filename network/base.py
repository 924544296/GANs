
'''
pass
'''


import argparse 






parser = argparse.ArgumentParser()
parser.add_argument("--path_image", type=str, default='celeba/img_align_celeba/', help="path of images")
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--dim_latent", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channel_g", type=int, default=64, help="number of generator channels")
parser.add_argument("--channel_d", type=int, default=64, help="number of discriminator channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--step_g", type=int, default=1, help="step of generator")
parser.add_argument("--path_work", type=str, default='runs/', help="path of images")
opt = parser.parse_args()
print(opt)



