import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, encoder_feature: int, decoder_hidden: int, attention_hidden: int):
        super().__init__()
        self.attn_hid = attention_hidden
        self.attn_enc_fc = nn.Linear(encoder_feature, attention_hidden)
        self.attn_dec_fc = nn.Linear(decoder_hidden, attention_hidden)
        self.attn_fc = nn.Linear(attention_hidden, 1)

    def forward(self, encoder_output: torch.Tensor, decoder_hidden: torch.Tensor) -> torch.Tensor:
        attn = torch.tanh(self.attn_enc_fc(encoder_output) + self.attn_dec_fc(decoder_hidden.permute(1, 0, 2)))
        attn = self.attn_fc(attn)
        attn = torch.softmax(attn, 1)
        return attn
