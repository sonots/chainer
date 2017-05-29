import numpy

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import reporter

from chainer.functions.connection import n_step_lstm as rnn


def permutate_list(lst, indices, inv):
    ret = [None] * len(lst)
    if inv:
        for i, ind in enumerate(indices):
            ret[ind] = lst[i]
    else:
        for i, ind in enumerate(indices):
            ret[i] = lst[ind]
    return ret


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0, force_tuple=True)
    return exs


class LSTMDecoder(chainer.ChainList):

    def __init__(self, n_layers, in_size, out_size, dropout=0.0):
        super(LSTMDecoder, self).__init__()
        self.n_layers = n_layers
        self.in_size = in_size
        self.out_size = out_size

        unit_size = in_size
        for l in range(n_layers):
            self.add_link(L.LSTM(unit_size, out_size))
            unit_size = out_size

        self.dropout = dropout


class Seq2seqAttention(chainer.Chain):

    """Implementaion of Luong's attentional NMT model."""

    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units):
        super(Seq2seqAttention, self).__init__(
            embed_x=L.EmbedID(n_source_vocab, n_units),
            embed_y=L.EmbedID(n_target_vocab, n_units),
            encoder=L.NStepLSTM(n_layers, n_units, n_units, 0.1),
            decoder=LSTMDecoder(n_layers, n_units * 2, n_units, 0.1)),
            W=L.Linear(n_units, n_target_vocab),
            Wc=L.Linear(n_units * 2, n_units),
            Wi=L.Linear(n_units, n_units)
        )

        self.n_layers = n_layers
        self.n_units = n_units

    def decode_once(self, h, h_tilde, source_hiddens, inf_mask):
        batch = h.shape[0]
        if h_tilde is None:
            h_tilde = chainer.Variable(
                self.xp.zeros((batch, self.n_units), 'f'))
        h = F.concat([h, h_tilde[:batch]], axis=1)

        ratio = self.decoder.dropout
        for layer in self.decoder:
            h = layer(F.dropout(h, ratio=ratio))
        h = F.dropout(h, ratio=ratio)

        s = F.batch_matmul(source_hiddens[:batch], h)
        s += inf[:batch, :, None]
        a = F.softmax(s)
        a = F.broadcast_to(a, source_hiddens[:batch].shape)
        wh = source_hiddens[:batch] * a
        cy = F.sum(wh, axis=1)

        h_tilde = F.tanh(self.Wc(F.concat([cy, h])))

        return h_tilde

    def __call__(self, *inputs):
        xs = inputs[:len(inputs) // 2]
        ys = inputs[len(inputs) // 2:]

        xs = [x[::-1] for x in xs]
        exs = sequence_embed(self.embed_x, xs)

        batch = len(xs)
        h, c, hxs = self.encoder(None, None, exs)

        # start decoding
        eos = self.xp.zeros(1, 'i')
        ys_in = [F.concat([eos, y], axis=0) for y in ys]
        eys = sequence_embed(self.embed_y, ys_in)

        indices = numpy.argsort([-y.shape[0] for y in ys]).astype('i')
        eys = permutate_list(eys, indices, inv=False)
        h = F.permutate(h, indices, axis=1)
        c = F.permutate(c, indices, axis=1)
        hxs = permutate_list(hxs, indices, inv=False)

        hxs_padded = F.pad_sequence(hxs, padding=0)
        inf = self.xp.zeros(hxs_padded.shape[0:2], 'f')
        for i, hx in enumerate(hxs):
            inf[i, hx.shape[0]:] = -1000

        # initialize decoder's first state
        for i, layer in enumerate(self.decoder):
            layer.reset_state()
            layer.h = h[i]
            layer.c = c[i]

        eys_t = F.transpose_sequence(eys)

        os = []
        h_tilde = None
        for i in range(len(eys_t)):
            h = eys_t[i]
            batch = h.shape[0]
            assert h.shape == (batch, self.n_units)

            h_tilde = self.decode_once(h, h_tilde, hxs_padded, inf)
            os.append(h_tilde)
        os = F.transpose_sequence(os)
        os = permutate_list(os, indices, inv=True)  # re-order
        concat_os = F.concat(os, axis=0)
        ys_out = [F.concat([y, eos], axis=0) for y in ys]  # reference symbols
        concat_ys_out = F.concat(ys_out, axis=0)

        batch = len(xs)
        loss = F.softmax_cross_entropy(
            self.W(concat_os), concat_ys_out, normalize=False) \
            * concat_ys_out.shape[0] / batch

        reporter.report({'loss': loss.data}, self)
        perp = self.xp.exp(loss.data / concat_ys_out.shape[0] * batch)
        reporter.report({'perp': perp}, self)
        return loss

    def translate(self, xs, max_length=50):
        batch = len(xs)
        with chainer.no_backprop_mode():
            xs = [x[::-1] for x in xs]
            exs = sequence_embed(self.embed_x, xs)
            h, c, hxs = self.encoder(None, None, exs, train=False)

            hxs_padded = F.pad_sequence(hxs, padding=0)

            inf = self.xp.zeros(hxs_padded.shape[0:2], 'f')
            for i, hx in enumerate(hxs):
                inf[i, hx.shape[0]:] = -1000

            for i, layer in enumerate(self.decoder):
                layer.reset_state()
                layer.h = h[i]
                layer.c = c[i]

            ys = self.xp.zeros(batch, 'i')  # eos
            eos = 0
            stop = self.xp.array([False for _ in range(batch)])

            result = []
            h_tilde = None
            for i in range(max_length):
                eys = self.embed_y(ys)
                assert eys.shape == (batch, self.n_units)
                h = eys
                h_tilde = self.decode_once(h, h_tilde, hxs_padded, inf)
                wy = self.W(h_tilde)
                ys = self.xp.argmax(wy.data, axis=1).astype('i')
                result.append(ys)

                stop = (ys == eos) | stop
                if self.xp.all(stop):
                    break

        result = cuda.to_cpu(self.xp.stack(result).T)

        # Remove EOS taggs
        outs = []
        for y in result:
            inds = numpy.argwhere(y == 0)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs
