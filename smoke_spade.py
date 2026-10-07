"""One-batch forward + tiny train step for SPADE (local sanity check)."""

import torch
from torch.utils.data import DataLoader

from gan_unet_model import PatchDiscriminator
from spade_model import SPADEGenerator
from train_gan_pix2pix import GANDataset, discriminator_loss, generator_adv_loss


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = SPADEGenerator(label_nc=4, ngf=64, img_h=256, img_w=512).to(device)
    D = PatchDiscriminator(in_ch=4, ndf=64).to(device)

    ds = GANDataset('cvpr_train_v2.csv', num_classes=4, max_samples=4)
    label, real = next(iter(DataLoader(ds, batch_size=2)))
    label, real = label.to(device), real.to(device)

    fake = G(label)
    assert fake.shape == real.shape, (fake.shape, real.shape)

    loss_D = discriminator_loss(D(label, real), D(label, fake.detach()))
    loss_G = generator_adv_loss(D(label, fake)) + torch.nn.functional.l1_loss(fake, real) * 10
    loss_D.backward()
    loss_G.backward()
    print('OK SPADE smoke:', float(loss_D), float(loss_G), 'device=', device)


if __name__ == '__main__':
    main()
