from typing import Tuple

import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        context_dim: int,
        encoder_dim: int,
        drop_prob: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self._init_hidden = nn.Parameter(torch.randn(self.num_layers * 1, 1, self.hidden_dim))

        self.enc2context_fc = nn.Linear(encoder_dim, context_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim + context_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.out_fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(
        self, inputs: torch.Tensor, hidden: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(inputs)
        out, hidden = self.rnn(torch.cat([emb, context], -1), hidden)
        out = self.out_fc(self.dropout(out))
        return out, hidden

    def make_context(self, encoder_features: torch.Tensor, attention_scores: torch.Tensor) -> torch.Tensor:
        glimps = (encoder_features * attention_scores).sum(1)
        return self.enc2context_fc(self.dropout(glimps)).unsqueeze(1)

    def get_init_hidden(self, batch_size: int) -> torch.Tensor:
        return self._init_hidden.repeat(1, batch_size, 1)
