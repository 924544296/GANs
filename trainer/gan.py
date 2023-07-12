import paddle as P 
import paddle.nn.functional as F 
import cv2 
from network.gan import Generator_GAN, Discriminator_GAN 
from dataloader.Dataset_ import Dataset_GAN 
from trainer.base import Trainer 


class Trainer_GAN(Trainer):
    #
    def __init__(self, opt):
        super().__init__(opt)
        self.net_g = Generator_GAN(opt.dim_latent, opt.channel_g)
        if opt.generate_only:
            self.net_g.set_state_dict(P.load(self.opt.path_result + 'net_g.pdparams'))
        else: 
            self.net_d = Discriminator_GAN(opt.channel_d)
            if self.opt.load_model:
                self.net_g.set_state_dict(P.load(self.opt.path_result + 'net_g.pdparams'))
                self.net_d.set_state_dict(P.load(self.opt.path_result + 'net_d.pdparams'))
            self.optimizer_g = P.optimizer.Adam(parameters=self.net_g.parameters(), 
                                learning_rate=2*opt.learning_rate, beta1=opt.beta1, beta2=opt.beta2)
            self.optimizer_d = P.optimizer.Adam(parameters=self.net_d.parameters(), 
                                learning_rate=opt.learning_rate, beta1=opt.beta1, beta2=opt.beta2)
            self.net_g.train()
            self.net_d.train()
            self.dataset = Dataset_GAN(opt.path_image)
    # 
    def trainer_d(self, image_real):
        #
        src_real = self.net_d(image_real)
        image_fake = self.net_g(P.randn([image_real.shape[0], self.opt.dim_latent, 1, 1]))
        src_fake = self.net_d(image_fake.detach())
        #
        loss = F.binary_cross_entropy_with_logits(src_real, P.ones_like(src_real)) + \
            F.binary_cross_entropy_with_logits(src_fake, P.zeros_like(src_fake))
        #
        self.optimizer_d.clear_grad()
        loss.backward()
        self.optimizer_d.step()
    #
    def trainer_g(self, image_real):
        #
        image_fake = self.net_g(P.randn([image_real.shape[0], self.opt.dim_latent, 1, 1]))
        src_fake = self.net_d(image_fake)
        #
        loss = F.binary_cross_entropy_with_logits(src_fake, P.ones_like(src_fake))
        #
        self.optimizer_g.clear_grad()
        loss.backward()
        self.optimizer_g.step()
    #
    def save_image(self, epoch, iteration): 
        self.net_g.eval() 
        image = self.net_g(P.randn([8, self.opt.dim_latent, 1, 1])) 
        image = image.reshape([2,4,3,64,64]).transpose([0,3,1,4,2]).reshape([128,256,3])
        image = (image[:,:,::-1]*127.5+127.5).numpy().astype('uint8')
        cv2.imwrite('%s Epoch %d__Iteration %d.png' % (self.opt.path_result+'image/', epoch, iteration), image)
        self.net_g.train() 
    #
    def train(self):
        super().train()