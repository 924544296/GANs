import paddle as P 
import paddle.nn.functional as F 
from paddle.io import DataLoader 


class Trainer:
    # 
    def __init__(self, opt):
        self.opt = opt 
    #
    def trainer_d(self):
        pass
    #
    def trainer_g(self):
        pass 
    #
    def train(self):
        #
        # net_d = Discriminator()
        # net_d.train()
        # net_g = Generator()
        # net_g.train()
        # optimizer_d = P.optimizer.Adam(parameters=net_d.parameters(), learning_rate=learning_rate, beta1=0.5)
        # optimizer_g = P.optimizer.Adam(parameters=net_g.parameters(), learning_rate=2*learning_rate, beta1=0.5)
        if self.opt.load_model:
            self.net_d.set_state_dict(P.load(self.opt.path_work + 'net_d.pdparams'))
            self.net_g.set_state_dict(P.load(self.opt.path_work + 'net_g.pdparams'))
        #
        iteration = 0
        for epoch in range(self.opt.epochs):
            dataloader = DataLoader(self.dataset, batch_size=self.opt.batch_size,
                            shuffle=True, num_workers=self.opt.n_cpu)
            for image in dataloader:
                self.trainer_d(image)
                iteration += 1
                if iteration % self.opt.step_g == 0:
                    self.trainer_g(image)
        #
                if iteration % 1000 == 0:
                    print('Epoch: ', epoch, ', Iteration: ', iteration)
                    P.save(self.net_d.state_dict(), self.opt.path_work + 'net_d.pdparams')
                    P.save(self.net_g.state_dict(), self.opt.path_work + 'net_g.pdparams')
                    # show(net_g)
                    self.net_g.train()
