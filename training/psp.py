from torch import nn
from . import psp_encoders


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class pSp(nn.Module):

    def __init__(self, encoder_type, c_dim, img_resolution, img_channels):
        super(pSp, self).__init__()
        self.encoder_type = encoder_type
        self.img_channels = img_channels
        
        # Define architecture
        self.encoder = self.set_encoder()
        # Load weights if needed

    def set_encoder(self):
        if self.encoder_type == 'gradual':
            encoder = psp_encoders.GradualStyleEncoder2(input_nc=self.img_channels)
        elif self.encoder_type == 'w':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', input_nc=self.img_channels)
        elif self.encoder_type == 'w+':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', input_nc=self.img_channels)
        else:
            raise Exception('{} is not a valid encoders'.format(self.encoder_type))
        return encoder

#     def load_weights(self):
#         if self.opts.checkpoint_path is not None:
#             print('Loading pSp from checkpoint: {}'.format(self.opts.checkpoint_path))
#             ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
#             self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
#             self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
#             self.__load_latent_avg(ckpt)
#         else:
#             print('Loading encoders weights from irse50!')
#             encoder_ckpt = torch.load(model_paths['ir_se50'])
#             # if input to encoder is not an RGB image, do not load the input layer weights
#             if self.opts.label_nc != 0:
#                 encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
#             self.encoder.load_state_dict(encoder_ckpt, strict=False)
#             print('Loading decoder weights from pretrained!')
#             ckpt = torch.load(self.opts.stylegan_weights)
#             self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
#             if self.opts.learn_in_w:
#                 self.__load_latent_avg(ckpt, repeat=1)
#             else:
#                 self.__load_latent_avg(ckpt, repeat=18)

    def forward(self, x, labels, input_code=False):
        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            # normalize with respect to the center of an average face
#             if self.opts.start_from_latent_avg:
#                 if self.opts.learn_in_w:
#                     codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
#                 else:
#                     codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)


#         if latent_mask is not None:
#             for i in latent_mask:
#                 if inject_latent is not None:
#                     if alpha is not None:
#                         codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
#                     else:
#                         codes[:, i] = inject_latent[:, i]
#                 else:
#                     codes[:, i] = 0
        return codes


#     def __load_latent_avg(self, ckpt, repeat=None):
#         if 'latent_avg' in ckpt:
#             self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
#             if repeat is not None:
#                 self.latent_avg = self.latent_avg.repeat(repeat, 1)
#         else:
#             self.latent_avg = None