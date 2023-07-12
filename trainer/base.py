import paddle as P 
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
    def save_image(self):
        pass 
    #
    def train(self):
        #
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
                if iteration % self.opt.sample_interval == 0:
                    print('Epoch: ', epoch, ', Iteration: ', iteration)
                    P.save(self.net_d.state_dict(), self.opt.path_result + 'net_d.pdparams')
                    P.save(self.net_g.state_dict(), self.opt.path_result + 'net_g.pdparams')
                    self.save_image(epoch, iteration)
                    self.net_g.train()
