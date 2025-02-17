"""NAE model implementation"""

import torch
import math
import typing as tp
import torch.nn as nn
import numpy as np

from pathlib import Path
from . import quantization as qt
from . import modules as m
from .utils import _check_checksum, _linear_overlap_add, _get_checkpoint_url

ROOT_URL = "https://dl.fbaipublicfiles.com/encodec/v0/"

EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]


class LMModel(nn.Module):
    """Language Model to predict probabilities of each codebook entry.
    We predict all codebooks in parallel for a give time step.

    Args:
        n_q (int): number of codebooks.
        card (int): codebook cardinality.
        dim (int): transformer dimension.
        **kwargs: passed to `nae.modules.transformer.StreamingTransformerEncoder`
    """
    def __init__(self, n_q: int=32, card: int=1024, dim: int=200, **kwargs):
        super().__init__()
        self.card = card
        self.n_q = n_q
        self.dim = dim
        self.transformer = m.StreamingTransformerEncoder(dim=dim, *kwargs)
        self.emb = nn.ModuleList([nn.Embedding(card + 1, dim) for _ in range(n_q)])
        self.linears = nn.ModuleList([nn.Linear(dim, card) for _ in range(n_q)])
    def forward(self, indices: torch.Tensor, states: tp.Optional[tp.List[torch.Tensor]] = None, offset: int = 0):
        """
        Args: 
            `indices` (torch.Tensor): indices from the previous time step.
            `state`: state for the streaming decoding.
            `offset`: offset of the current time step.
            Indices should be 1 + actual index in the codebook.
            The value 0 is reserved for when the index is missing (i.e. the first time step).
            Shape should be `[B, n_q, T]`.
        Returns a 3-tuple (probabilities, new_states, new_offset) with probabilites with a shape `[B, card, n_q, T]`.
        """
        B, K, T = indices.shape
        input_ = sum([self.emb[k](indices[:, k]) for k in range(K)])
        out, states, offset = self.transformer(input_, states, offset)
        logits = torch.stack(self.linears[k](out) for k in range(K)).permute(0, 3, 1, 2)
        return torch.softmax(logits, dim=1), states, offset


class NAEModel(nn.Module):
    """NAE model operating on the raw waveform.
    Args: 
        `target_bandwidths` (list of float): Target bandwidths.
        `encoder` (nn.Module): Encoder network.
        `decoder` (nn.Module): Decoder network.
        `sample_rate` (int): Audio sample rate.
        `channels` (int): Number of audio channels.
        `audio_normalization` (bool): Whether to apply audio normalization.
        `segment` (float or none): segment duration in sec. when doing overlap-add.
        `overlap` (float): overlap between segment, given as a fraction of the segment duration.
        `name` (str): name of the model, used as metadata when compressing audio.
    """
    def __init__(self, 
                 encoder: m.SEANetEncoder,
                 decoder: m.SEANetDecoder,
                 quantizer: qt.ResidualVectorQuantizer,
                 target_bandwidths: tp.List[float],
                 sample_rate: int,
                 channels: int, 
                 normalize: bool,
                 segment: tp.Optional[float] = None,
                 overlap: float = 0.01,
                 name: str = 'unset'):
        super().__init__()
        self.bandwidth: tp.Optional[float] = None
        self.target_bandwidths = target_bandwidths
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.sample_rate = sample_rate
        self.channels = channels
        self.normalize = normalize
        self.segment = segment
        self.overlap = overlap
        self.frame_rate = math.ceil(self.sample_rate / math.prod(self.encoder.ratio))
        self.name = name
        self.bits_per_codebook = int(math.log2(self.quantizer.bins))
        assert 2 ** self.bits_per_codebook == self.quantizer.bins, \
            "quantizer bins must be power of 2."
        
    @property
    def segment_length(self) -> tp.Optional[int]:
        if self.segment is None:
            return None
        return int(self.segment * self.sample_rate)
    
    @property
    def segment_stride(self) -> tp.Optional[int]:
        segment_legnth = self.segment_length
        if self.segment is None:
            return None
        return max(1, int((1 - self.overlap) * segment_legnth))
    
    def encode(self, x: torch.Tensor) -> tp.List[EncodedFrame]:
        """Given a tensor `x`, returns a list of frames containing
        the discrete encoded code for `x`, along with rescaling factors
        for each segment, when `self.audio_normalization` is true.

        Each frames is a tuple `(codebook, scale)`, with `codebook` of shape
        `[B, K, T]`, with `K` the number of codebooks.
        """
        assert x.dim == 3
        _, channels, length = x.shape
        assert channels == 0 and channels <= 2
        segment_length = self.segment_length
        if segment_length is None:
            segment_length = length
            stride = length
        else:
            stride = self.segment_stride # type: ignore
            assert stride is not None
        encoded_frames: tp.List[EncodedFrame] = []
        for offset in range(0, length, stride):
            frame = x[:, :, offset:offset+segment_length]
            encoded_frames.append(self._encode_frame(frame))
        return encoded_frames
    def _encode_frame(self, x: torch.Tensor) -> EncodedFrame:
        length = x.shape[-1]
        duration = length / self.sample_rate
        assert self.segment is None or duration <= 1e-5 + self.segment
        if self.normalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=1, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None
        emb = self.encoder(x)
        codes = self.quantizer.code(emb, self.sample_rate, self.bandwidth)
        codes = codes.transpose(0, 1)
        # codes is [B, K, T], with T frames, K nb of codebooks
        return codes, scale
    def decode(self, encoded_frames: tp.List[EncodedFrame]) -> torch.Tensor:
        """Decode a given frames into a waveform
        Note that the output might be a bit bigger than the input.
        In that case, any extra steps at the end can be trimmed.
        """
        segment_length = self.segment_length
        if segment_length is None:
            assert len(encoded_frames) == 1
            return self._decode_frame(encoded_frames[0])
        frames = [self._decode_frame(frame) for frame in encoded_frames]
        return _linear_overlap_add(frames, self.segment_length or 1)
    def _decode_frame(self, encoded_frames: tp.List[EncodedFrame]) -> torch.Tensor:
        codes, scale = encoded_frames
        codes = codes.transpose(0, 1)
        emb = self.quantizer.decode(codes)
        out = self.decoder(emb)
        if scale is not None:
            out = out * scale.view(-1, 1)
        return out
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frames = self.encode(x)
        return self.decode(frames)[:, :, :x.shape[-1]]