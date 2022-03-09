import torch 
from torch import nn 
from .transformer import Encoder, Decoder, ScaledDotProductAttentionMemory, MeshedDecoder
from .containers import Module

class AICModel(Module): 
    def __init__(self, config):
        super(AICModel, self).__init__() 
        self.model_d = config.n_embd 
        self.clip_dim = config.clip_dim 
        self.clip_length = config.clip_length
        self.feature_project = nn.Linear(config.clip_dim, config.clip_length*config.n_embd) 
        self.visual_encoder = Encoder(config.n_layer, config.clip_length, config.n_embd) 
        self.language_decoder = Decoder(config.vocab_size) 

        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights() 
    
    def init_weights(self):
        for p in self.visual_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.language_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    def forward(self, images, seq):
        images = self.feature_project(images).view(-1, self.clip_length, self.clip_dim)
        enc_output, mask_enc = self.visual_encoder(images)
        dec_output = self.language_decoder(seq, enc_output, mask_enc)
        return dec_output



