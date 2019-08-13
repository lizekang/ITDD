""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.

        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder(src, lengths)
        self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths)

        return dec_out, attns


class ConvModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(ConvModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, knl, src, tgt, src_lengths, knl_lengths):
        tgt = tgt[:-1]
        enc_state, memory_bank, lengths = self.encoder(src, src_lengths)
        knowledge_encoding, _, _ = self.encoder(knl, knl_lengths)
        self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      knowledge_encoding=knowledge_encoding,
                                      memory_lengths=src_lengths)
        return dec_out, attns


class KTransformerModel(nn.Module):
    def __init__(self, encoder, decoder, decoder2):
        super(KTransformerModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decoder2 = decoder2

    def forward(self, knl, src, tgt, src_lengths, knl_lengths):
        tgt = tgt[:-1]
        enc_state, his_bank, src_bank, knl_bank, lengths = self.encoder(src, knl, src_lengths, knl_lengths)
        self.decoder.init_state(src[100:, :, :], src[100:, :, :], src_bank, enc_state)
        first_dec_out, first_attns = self.decoder(tgt, src_bank, his_bank,
                                      memory_lengths=None)
        # log_probs [tgt_len, batch_size, vocab_size]
        first_log_probs = self.generator(first_dec_out.squeeze(0))
        _, first_dec_words = torch.max(first_log_probs, 2)
        first_dec_words = first_dec_words.unsqueeze(2)
        self.decoder2.init_state(first_dec_words, knl[600:, :, :], None, None)
        emb, decode1_bank, decode1_mask = self.encoder.histransformer(first_dec_words, None)
        second_dec_out, second_attns = self.decoder2(tgt, decode1_bank, knl_bank, memory_lengths=None)
        
        return first_dec_out, first_attns, second_dec_out, second_attns

