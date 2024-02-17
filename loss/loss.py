from loss import discriminator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.Gradient import gradient 

class L1_Charbonnier_loss(nn.Module):
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, sr, hr):
        diff = torch.add(sr, -hr)
        error = torch.sqrt( diff * diff + self.eps)
        loss = torch.sum(error)
        return loss

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.loss = nn.L1Loss()
    def forward(self, sr, hr):
        return self.loss(sr, hr)
    
    
class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.gradient = gradient()
        self.loss = nn.L1Loss()

    def forward(self, sr, hr):
        sr_g = self.gradient((sr+1) /2)
        hr_g = self.gradient((hr+1) /2)
        return  self.loss(sr_g, hr_g)
    
    
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        
    def forward(self, sr_vgg, hr_vgg):
        return F.mse_loss(sr_vgg, hr_vgg)

class AdversarialLoss(nn.Module):
    def __init__(self, logger, use_cpu = False, gan_type = 'WGAN_GP', gan_k = 1,  lr_dis = 1e-4, img_size = 160):
        super(AdversarialLoss, self).__init__()
        self.logger        = logger
        self.gan_type      = gan_type
        self.gan_k         = gan_k
        self.device        = torch.device('cpu' if use_cpu else 'cuda')
        # self.discriminator = discriminator.Discriminator(img_size).to(self.device)
        self.discriminator = discriminator.Discriminator().to(self.device)

        if (gan_type in ['WGAN_GP', 'GAN']):
            self.optimizer = optim.Adam(
                self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-8, lr=lr_dis
            )
        else:
            raise SystemExit('Error: no such type of GAN!')

        self.bce_loss = torch.nn.BCELoss().to(self.device)

        # if (D_path):
        #     self.logger.info('load_D_path: ' + D_path)
        #     D_state_dict = torch.load(D_path)
        #     self.discriminator.load_state_dict(D_state_dict['D'])
        #     self.optimizer.load_state_dict(D_state_dict['D_optim'])
            
    def forward(self, fake, real):
        fake_detach = fake.detach()

        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            d_fake = self.discriminator(fake_detach)
            d_real = self.discriminator(real)
            if (self.gan_type.find('WGAN') >= 0):
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand(real.size(0), 1, 1, 1).to(self.device)
                    epsilon = epsilon.expand(real.size())
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty

            elif (self.gan_type == 'GAN'):
                valid_score = torch.ones(real.size(0), 1).to(self.device)
                fake_score = torch.zeros(real.size(0), 1).to(self.device)
                real_loss = self.bce_loss(torch.sigmoid(d_real), valid_score)
                fake_loss = self.bce_loss(torch.sigmoid(d_fake), fake_score)
                loss_d = (real_loss + fake_loss) / 2.

            # Discriminator update
            loss_d.backward()
            self.optimizer.step()

        d_fake_for_g = self.discriminator(fake)
        if (self.gan_type.find('WGAN') >= 0):
            loss_g = -d_fake_for_g.mean()
        elif (self.gan_type == 'GAN'):
            loss_g = self.bce_loss(torch.sigmoid(d_fake_for_g), valid_score)

        # Generator loss
        return loss_g
  
    def state_dict(self):
        D_state_dict = self.discriminator.state_dict()
        D_optim_state_dict = self.optimizer.state_dict()
        return D_state_dict, D_optim_state_dict

def get_loss_dict(args, logger):
    loss = {}

    loss['rec_loss'] = ReconstructionLoss()
    loss['grd_loss'] = GradientLoss()
    loss['per_loss'] = PerceptualLoss()
    loss['adv_loss'] = AdversarialLoss(logger = logger, use_cpu = args.cpu, gan_k = args.GAN_k, 
                                       lr_dis = args.lr_rate_dis, gan_type = args.GAN_type,  img_size = args.img_training_size)
    return loss
