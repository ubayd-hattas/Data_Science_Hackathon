"""A small triplet-loss embedding network, in pure numpy.

``METRIC_LEARNING_APPROACH.md`` specifies training an embedding ``f: R^60 -> R^d``
on Madrid with triplet loss, then classifying Amsterdam by nearest prototype in
that embedding space. Notebook 4 skips the embedding and runs prototypes on the
raw standardised features, so nothing Madrid-trained is actually carried across
cities. This module supplies the missing piece.

No torch / tensorflow dependency: the network is a 3-layer MLP with hand-written
forward and backward passes and an Adam optimiser. At 60 inputs and a few
thousand parameters this trains in seconds on CPU, and it runs anywhere numpy
does (the hackathon hub included).

Typical use::

    from src.embedding import TripletEmbedding
    emb = TripletEmbedding(dim=16, seed=42).fit(X_madrid, y_madrid)
    Z_madrid    = emb.transform(X_madrid)
    Z_amsterdam = emb.transform(X_amsterdam)      # prototypes computed here
"""

from __future__ import annotations

import numpy as np

__all__ = ["TripletEmbedding"]


def _he_init(rng: np.random.Generator, n_in: int, n_out: int) -> np.ndarray:
    return rng.standard_normal((n_in, n_out)).astype(np.float32) * np.sqrt(2.0 / n_in)


class TripletEmbedding:
    """MLP embedding trained with a margin triplet loss.

    Architecture: ``in -> hidden[0] -> ReLU -> hidden[1] -> ReLU -> dim``, then
    L2-normalisation so every embedding lies on the unit hypersphere. That keeps
    triplet distances bounded and training stable, as the approach note
    recommends.

    Args:
        dim: embedding dimensionality.
        hidden: the two hidden-layer widths.
        margin: triplet margin in squared-distance units on the unit sphere
            (meaningful range roughly 0-4).
        lr: Adam learning rate.
        epochs: passes over the mined triplet set.
        batch_size: triplets per optimiser step.
        triplets_per_anchor: triplets mined per anchor each epoch.
        semi_hard: if True, prefer negatives currently closer than the positive
            plus margin (the informative ones), falling back to random.
        standardize: if True, fit an internal mean/std on the training X and
            reapply it in ``transform``. Leave True unless X is already scaled.
        seed: RNG seed.
    """

    def __init__(
        self,
        dim: int = 16,
        hidden: tuple[int, int] = (64, 32),
        margin: float = 0.5,
        lr: float = 1e-3,
        epochs: int = 60,
        batch_size: int = 256,
        triplets_per_anchor: int = 5,
        semi_hard: bool = True,
        standardize: bool = True,
        seed: int = 42,
    ) -> None:
        self.dim = dim
        self.hidden = hidden
        self.margin = float(margin)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.triplets_per_anchor = int(triplets_per_anchor)
        self.semi_hard = bool(semi_hard)
        self.standardize = bool(standardize)
        self.seed = int(seed)
        self.history_: list[float] = []

    # -- network plumbing --------------------------------------------------

    def _init_params(self, n_features: int) -> None:
        rng = np.random.default_rng(self.seed)
        sizes = [n_features, self.hidden[0], self.hidden[1], self.dim]
        self.W = [_he_init(rng, sizes[i], sizes[i + 1]) for i in range(3)]
        self.b = [np.zeros(sizes[i + 1], dtype=np.float32) for i in range(3)]
        self._mW = [np.zeros_like(w) for w in self.W]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(x) for x in self.b]
        self._vb = [np.zeros_like(x) for x in self.b]
        self._t = 0

    def _forward(self, X: np.ndarray):
        h0 = X @ self.W[0] + self.b[0]
        a0 = np.maximum(h0, 0.0)
        h1 = a0 @ self.W[1] + self.b[1]
        a1 = np.maximum(h1, 0.0)
        z = a1 @ self.W[2] + self.b[2]
        norm = np.sqrt((z * z).sum(axis=1, keepdims=True) + 1e-12)
        e = z / norm
        return e, (X, a0, h0, a1, h1, z, norm)

    def _backward(self, grad_e: np.ndarray, cache) -> None:
        X, a0, h0, a1, h1, z, norm = cache
        n = X.shape[0]

        # through e = z / ||z||
        dot = (grad_e * (z / norm)).sum(axis=1, keepdims=True)
        grad_z = (grad_e - (z / norm) * dot) / norm

        gW2 = a1.T @ grad_z / n
        gb2 = grad_z.mean(axis=0)
        da1 = grad_z @ self.W[2].T
        dh1 = da1 * (h1 > 0)

        gW1 = a0.T @ dh1 / n
        gb1 = dh1.mean(axis=0)
        da0 = dh1 @ self.W[1].T
        dh0 = da0 * (h0 > 0)

        gW0 = X.T @ dh0 / n
        gb0 = dh0.mean(axis=0)

        self._adam_step([gW0, gW1, gW2], [gb0, gb1, gb2])

    def _adam_step(self, gW, gb, beta1=0.9, beta2=0.999, eps=1e-8) -> None:
        self._t += 1
        bc1 = 1.0 - beta1 ** self._t
        bc2 = 1.0 - beta2 ** self._t
        for i in range(3):
            self._mW[i] = beta1 * self._mW[i] + (1 - beta1) * gW[i]
            self._vW[i] = beta2 * self._vW[i] + (1 - beta2) * (gW[i] ** 2)
            self.W[i] -= self.lr * (self._mW[i] / bc1) / (np.sqrt(self._vW[i] / bc2) + eps)
            self._mb[i] = beta1 * self._mb[i] + (1 - beta1) * gb[i]
            self._vb[i] = beta2 * self._vb[i] + (1 - beta2) * (gb[i] ** 2)
            self.b[i] -= self.lr * (self._mb[i] / bc1) / (np.sqrt(self._vb[i] / bc2) + eps)

    # -- triplet mining --------------------------------------------------

    def _mine(self, e: np.ndarray, y: np.ndarray, rng: np.random.Generator):
        """Vectorised (anchor, positive, negative) mining for the whole set.

        Positives are drawn within class, negatives across classes, grouped by
        class so the work is four array ops rather than a per-anchor loop. With
        ``semi_hard`` the negative is picked from a small candidate pool as the
        one whose distance sits just beyond the positive (the informative case),
        falling back to the hardest available.
        """
        n = len(y)
        k = self.triplets_per_anchor
        classes = np.unique(y)
        idx_by_class = {int(c): np.where(y == c)[0] for c in classes}

        anc = np.repeat(np.arange(n), k)
        y_anc = y[anc]
        pos = np.empty(n * k, dtype=np.int64)
        neg = np.empty(n * k, dtype=np.int64)

        for c in classes:
            m = y_anc == c
            cnt = int(m.sum())
            if cnt == 0:
                continue
            same = idx_by_class[int(c)]
            diff = np.where(y != c)[0]

            if len(same) >= 2:
                p = rng.choice(same, size=cnt)
                a_sub = anc[m]
                for _ in range(4):                     # resample self-matches
                    bad = p == a_sub
                    if not bad.any():
                        break
                    p[bad] = rng.choice(same, size=int(bad.sum()))
            else:
                p = np.full(cnt, same[0])
            pos[m] = p

            if self.semi_hard and len(diff) > 0:
                pool_n = min(8, len(diff))
                cand = rng.choice(diff, size=(cnt, pool_n))
                a_sub = anc[m]
                d_ap = ((e[a_sub] - e[pos[m]]) ** 2).sum(axis=1)
                # squared dist anchor -> each candidate, no (cnt,pool,dim) tensor
                sq = (e * e).sum(axis=1)
                d_an = (sq[a_sub][:, None] + sq[cand]
                        - 2.0 * np.einsum("id,ipd->ip", e[a_sub], e[cand]))
                ok = (d_an > d_ap[:, None]) & (d_an < d_ap[:, None] + self.margin)
                pick = np.where(ok.any(axis=1), ok.argmax(axis=1), d_an.argmin(axis=1))
                neg[m] = cand[np.arange(cnt), pick]
            else:
                neg[m] = rng.choice(diff, size=cnt)

        return anc, pos, neg

    # -- public API --------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TripletEmbedding":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if self.standardize:
            self._mu = X.mean(axis=0)
            self._sd = X.std(axis=0) + 1e-8
            Xs = (X - self._mu) / self._sd
        else:
            Xs = X

        self._init_params(Xs.shape[1])
        rng = np.random.default_rng(self.seed)

        for _ in range(self.epochs):
            e_all, _ = self._forward(Xs)
            a_idx, p_idx, n_idx = self._mine(e_all, y, rng)
            order = rng.permutation(len(a_idx))
            a_idx, p_idx, n_idx = a_idx[order], p_idx[order], n_idx[order]

            epoch_loss, n_batches = 0.0, 0
            for s in range(0, len(a_idx), self.batch_size):
                bi = slice(s, s + self.batch_size)
                rows, inv = np.unique(
                    np.concatenate([a_idx[bi], p_idx[bi], n_idx[bi]]),
                    return_inverse=True,
                )
                m = len(a_idx[bi])
                ia, ip, in_ = inv[:m], inv[m:2 * m], inv[2 * m:]

                e, cache = self._forward(Xs[rows])
                d_ap = ((e[ia] - e[ip]) ** 2).sum(axis=1)
                d_an = ((e[ia] - e[in_]) ** 2).sum(axis=1)
                viol = d_ap - d_an + self.margin
                active = viol > 0
                epoch_loss += float(np.maximum(viol, 0).mean())
                n_batches += 1
                if not active.any():
                    continue

                grad_e = np.zeros_like(e)
                scale = 2.0 / m
                np.add.at(grad_e, ia[active], scale * (e[in_][active] - e[ip][active]))
                np.add.at(grad_e, ip[active], scale * (e[ip][active] - e[ia][active]))
                np.add.at(grad_e, in_[active], scale * (e[ia][active] - e[in_][active]))
                self._backward(grad_e, cache)

            self.history_.append(epoch_loss / max(n_batches, 1))

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.standardize:
            X = (X - self._mu) / self._sd
        return self._forward(X)[0]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)
