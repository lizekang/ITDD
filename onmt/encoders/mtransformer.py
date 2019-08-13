"""
Implementation of "Attention is All You Need"
"""
import torch
import torch.nn as nn

import onmt
from onmt.encoders.encoder import EncoderBase
# from onmt.utils.misc import aeq
from onmt.modules.position_ffn import PositionwiseFeedForward


class STransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(STransformerEncoderLayer, self).__init__()

        self.self_attn = onmt.modules.MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class ATransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(ATransformerEncoderLayer, self).__init__()

        self.self_attn = onmt.modules.MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.knowledge_attn = onmt.modules.MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.context_attn = onmt.modules.MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, src_mask, knl_bank, knl_mask, his_bank, his_mask):
        """
        Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
        """
        input_norm = self.layer_norm_1(inputs)
        query, _ = self.self_attn(input_norm, input_norm, input_norm,
                                  mask=src_mask)
        query = self.dropout(query) + inputs
        query_norm = self.layer_norm_2(query)
        knl_out, _ = self.knowledge_attn(knl_bank, knl_bank, query_norm,
                                         mask=knl_mask)
        knl_out = self.dropout(knl_out) + query
        if his_bank is not None:
            his_bank = his_bank.transpose(0, 1).contiguous()
            knl_out_norm = self.layer_norm_3(knl_out)

            out, _ = self.context_attn(his_bank, his_bank, knl_out_norm,
                                       mask=his_mask)
            out = self.dropout(out) + knl_out
            return self.feed_forward(out)
        else:
            return self.feed_forward(knl_out)


class KNLTransformerEncoder(EncoderBase):
    """
    The Transformer encoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings

    Returns:
        (`FloatTensor`, `FloatTensor`):

        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings):
        super(KNLTransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [STransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src, lengths=None):
        """ See :obj:`EncoderBase.forward()`"""
        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()
        words = src[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]
        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            out = self.transformer[i](out, mask)
        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous(), mask


class HTransformerEncoder(EncoderBase):
    """
    The Transformer encoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings

    Returns:
        (`FloatTensor`, `FloatTensor`):

        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings):
        super(HTransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [ATransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src, history_bank=None, knl_bank=None, knl_mask=None, his_mask=None):
        """ See :obj:`EncoderBase.forward()`"""

        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()
        words = src[:, :, 0].transpose(0, 1)
        padding_idx = self.embeddings.word_padding_idx
        src_mask = words.data.eq(padding_idx).unsqueeze(1)
        # if history_bank is None:
        #     history_bank = torch.randn(out.size()).cuda().transpose(0, 1).contiguous()
        # temp = torch.cat([out, history_bank.transpose(0, 1).contiguous()], 2)
        # out = out + self.w(temp)

        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            out = self.transformer[i](out, src_mask, knl_bank, knl_mask, history_bank, his_mask)
        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous(), src_mask


class TransformerEncoder(EncoderBase):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings):
        # KTransformerEncoder 与 HTransformerEncoder暂时共享embedding
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.embeddings = embeddings
        self.knltransformer = KNLTransformerEncoder(num_layers, d_model, heads,
                                                    d_ff, dropout, embeddings)
        self.histransformer = KNLTransformerEncoder(num_layers, d_model, heads,
                                                    d_ff, dropout, embeddings)
        self.htransformer = HTransformerEncoder(num_layers, d_model, heads,
                                                d_ff, dropout, embeddings)

    def forward(self, src, knl=None, lengths=None, knl_lengths=None):
        history = self.history2list(src, knl)
        tgt_knl = knl[600:, :, :]
        emb, knl_bank_tgt, knl_mask = self.knltransformer(tgt_knl, None)
        emb, src_bank, src_mask = self.histransformer(src[100:, :, :], None)
        his_bank = None
        his_mask = None
        for h in history:
            u = h[0]
            k = h[1]
            emb, knl_bank, knl_mask = self.knltransformer(k, None)
            knl_bank_input = knl_bank.transpose(0, 1).contiguous()
            emb, his_bank, his_mask = self.htransformer(u, his_bank, knl_bank_input, knl_mask, his_mask)
        return emb, his_bank, src_bank, knl_bank_tgt, lengths

    @staticmethod
    def history2list(src, knl):
        u1 = src[:50, :, :]
        u2 = src[50:100, :, :]
        u3 = src[100:, :, :]
        k1 = knl[:200, :, :]
        k2 = knl[200:400, :, :]
        k3 = knl[400:600, :, :]
        return (u1, k1), (u2, k2), (u3, k3)
