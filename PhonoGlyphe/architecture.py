import torch

import torch.nn as nn
from torch import sqrt, cos, sin, exp, log, pi



class InputEmbeddings(nn.Module):

    def __init__(self, vocab_size, dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.stabilisation_coef = sqrt(torch.tensor(self.dim))

        self.embedding = nn.Embedding(vocab_size, dim)

    def forward(self, x):
        return self.stabilisation_coef * self.embedding(x)


class PositionalEmbeddings(nn.Module):

    def __init__(self, context_length, dim):
        super().__init__()

        if type(context_length) != torch.Tensor:
            context_length = torch.tensor(context_length)

        if type(dim) != torch.Tensor:
            dim = torch.tensor(dim)

        self.context_length = context_length
        self.dim = dim

        pos_embedding = torch.empty((context_length, dim), dtype=torch.float, requires_grad=False)

        dims = torch.arange(1, dim+1, dtype=torch.float)
        positions = torch.arange(1, context_length+1, dtype=torch.float)[:, None]

        encoding_base = (2*pi) * exp(log(context_length) * -(dims/dim))

        pos_embedding[0::2] = sin(positions[0::2] * encoding_base)
        pos_embedding[1::2] = cos(positions[1::2] * encoding_base)

        self.register_buffer("pos_embedding", pos_embedding[None, :, :])

    def forward(self, x):
        return x + self.pos_embedding


class NormalisationLayer(nn.Module):

    def __init__(self, dim, epsilon):
        super().__init__()

        self.layer_norm = nn.LayerNorm(dim, eps=epsilon)

    def forward(self, x):
        return self.layer_norm(x)


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, context_length, dim, head_count, abstraction_coef, dropout):
        super().__init__()
        self.context_length = context_length
        self.dim = dim
        self.head_count = head_count
        self.abstract_dim = int(dim * abstraction_coef)
        self.abstract_dim_sqrt = sqrt(torch.tensor(self.abstract_dim))
        self.mask_coef = torch.tensor(-1e12)

        self.q_weights = nn.Linear(dim, head_count*self.abstract_dim, bias=True)
        self.k_weights = nn.Linear(dim, head_count*self.abstract_dim, bias=True)
        self.v_weights = nn.Linear(dim, head_count*self.abstract_dim, bias=True)

        self.p_weights = nn.Linear(head_count*self.abstract_dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Transformation linéaire
        q_vect = self.q_weights(q)
        k_vect = self.k_weights(k)
        v_vect = self.v_weights(v)

        # Réorganisation pour parallélisme
        q_vect = q_vect.view(batch_size, self.context_length, self.head_count, self.abstract_dim).permute(0, 2, 1, 3)
        k_vect = k_vect.view(batch_size, self.context_length, self.head_count, self.abstract_dim).permute(0, 2, 1, 3)
        v_vect = v_vect.view(batch_size, self.context_length, self.head_count, self.abstract_dim).permute(0, 2, 1, 3)

        # Calcul des scores d'attention
        attention_scores = torch.matmul(q_vect, k_vect.transpose(-2, -1)) / self.abstract_dim_sqrt
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask==0, self.mask_coef)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Application des poids d'attention
        attention_output = torch.matmul(attention_weights, v_vect)

        # Concaténation des résultats
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, self.context_length, self.head_count*self.abstract_dim)

        # Projection de sortie
        output = self.p_weights(attention_output)

        return output


class FeedForwardBlock(nn.Module):

    def __init__(self, dim, abstraction_coef, dropout):
        super().__init__()
        abstract_dim = int(dim * abstraction_coef)

        self.block = nn.Sequential(
            nn.Linear(dim, abstract_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(abstract_dim, dim, bias=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualConnectionGroup(nn.Module):

    def __init__(self, norm_params, dropout):
        super().__init__()
        self.norm = NormalisationLayer(*norm_params)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, layer):
        return x + self.dropout(layer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self, norm_params, attention_params, feed_forward_params, dropout):
        super().__init__()
        self.rcg_1 = ResidualConnectionGroup(norm_params, dropout)
        self.attn = MultiHeadAttentionBlock(*attention_params, dropout)
        self.rcg_2 = ResidualConnectionGroup(norm_params, dropout)
        self.ffw = FeedForwardBlock(*feed_forward_params, dropout)

    def forward(self, x, mask):
        attn_call = lambda x: self.attn(x, x, x, mask)
        ffw_call = lambda x: self.ffw(x)

        x = self.rcg_1(x, attn_call)
        x = self.rcg_2(x, ffw_call)
        return x


class Encoder(nn.Module):

    def __init__(self, encoder_block_count, encoder_block_params, norm_params, dropout):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([EncoderBlock(*encoder_block_params, dropout) for _ in range(encoder_block_count)])
        self.norm = NormalisationLayer(*norm_params)

    def forward(self, x, mask):
        for block in self.encoder_blocks:
            x = block(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(self, norm_params, self_attention_params, cross_attention_params, feed_forward_params, dropout):
        super().__init__()
        self.rcg_1 = ResidualConnectionGroup(norm_params, dropout)
        self.self_attn = MultiHeadAttentionBlock(*self_attention_params, dropout)
        self.rcg_2 = ResidualConnectionGroup(norm_params, dropout)
        self.cross_attn = MultiHeadAttentionBlock(*cross_attention_params, dropout)
        self.rcg_3 = ResidualConnectionGroup(norm_params, dropout)
        self.ffw = FeedForwardBlock(*feed_forward_params, dropout)

    def forward(self, x, y, mask):
        self_attn_call = lambda x: self.self_attn(x, x, x, mask)
        cross_attn_call = lambda x: self.cross_attn(x, y, y, mask)
        ffw_call = lambda x: self.ffw(x)

        x = self.rcg_1(x, self_attn_call)
        x = self.rcg_2(x, cross_attn_call)
        x = self.rcg_3(x, ffw_call)
        return x


class Decoder(nn.Module):

    def __init__(self, decoder_block_count, decoder_block_params, norm_params, dropout):
        super().__init__()
        self.decoder_blocks = nn.ModuleList([DecoderBlock(*decoder_block_params, dropout) for _ in range(decoder_block_count)])
        self.norm = NormalisationLayer(*norm_params)

    def forward(self, x, y, mask):
        for block in self.decoder_blocks:
            x = block(x, y, mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, context_length, dim, vocab_size):
        super().__init__()
        self.proj1 = nn.Linear(context_length, 1)
        self.proj2 = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = x.transpose(-2, -1)
        x = self.proj1(x).squeeze(-1)
        return self.proj2(x).log_softmax(dim=-1)


class Transformer(nn.Module):

    def __init__(self, source_embeddings_params, target_embeddings_params, positional_embeddings_params, encoder_params, decoder_params, projection_params, dropout):
        super().__init__()
        self.source_embeddings = InputEmbeddings(*source_embeddings_params)
        self.target_embeddings = InputEmbeddings(*target_embeddings_params)
        self.positional_embeddings = PositionalEmbeddings(*positional_embeddings_params)
        self.encoder = Encoder(*encoder_params, dropout)
        self.decoder = Decoder(*decoder_params, dropout)
        self.projection = ProjectionLayer(*projection_params)

    def encode(self, x, mask):
        x = self.source_embeddings(x)
        x = self.positional_embeddings(x)
        return self.encoder(x, mask)

    def decode(self, x, y, mask):
        x = self.target_embeddings(x)
        x = self.positional_embeddings(x)
        return self.decoder(x, y, mask)

    def project(self, x):
        return self.projection(x)