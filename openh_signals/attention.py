"""Detached two-pass, full-denominator measurements at the rotated Q/K boundary."""

import hashlib
import math

import torch


def _groups(rows, length, groups, device):
    if rows.dtype != torch.long or rows.device != device or rows.shape != (length,):
        raise ValueError("group layout must label the physical axis")
    if not 1 <= groups <= 12 or not ((rows >= 0) & (rows < groups)).all():
        raise ValueError("group outside the v1 partition")
    return torch.nn.functional.one_hot(rows, groups).float()


@torch.no_grad()
def attention_summary(
    q,
    k,
    v,
    qgroup,
    kgroup,
    n_groups,
    *,
    causal=False,
    offset=0,
    scale=None,
    tile=64,
    calibration=False,
    digest=False,
    visit=None,
):
    """Observe one sample/pass; no padded keys. Optional tile consumer is observer-only.

    Mixture is returned only for an explicit calibration. A missing query group is NaN,
    whereas a blocked key group has exactly zero mass and zero legal pairs.
    """
    nq, h, d = q.shape
    nk, hk, dk = k.shape
    if (
        not min(nq, nk, hk, h, d, tile) > 0
        or d != dk
        or k.shape != v.shape
        or h % hk
        or q.device != k.device
        or v.device != k.device
    ):
        raise ValueError("invalid grouped-query dimensions")
    if scale is None:
        scale = 1 / math.sqrt(d)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid attention scale")
    # Never flip backend flags on the live process. Refuse reduced precision observer GEMMs.
    if q.is_cuda and torch.get_float32_matmul_precision() != "highest":
        raise ValueError("FP32 observer requires IEEE matmul precision")
    with torch.autocast(device_type=q.device.type, enabled=False):
        qhot = _groups(qgroup, nq, n_groups, q.device)
        khot = _groups(kgroup, nk, n_groups, k.device)
        counts = qhot.sum(0).long()
        mass = qhot.new_zeros((h, n_groups, n_groups))
        ent = qhot.new_zeros((h, n_groups))
        legal = torch.zeros((n_groups, n_groups), device=q.device, dtype=torch.long)
        result = {"counts": counts, "legal": legal}
        if calibration:
            result["mixture"] = qhot.new_empty(q.shape)
        sha = hashlib.sha256(repr((nq, nk, h, tile)).encode()) if digest else None
        kid = torch.arange(h, device=q.device) // (h // hk)
        error = qhot.new_zeros(())
        for i in range(0, nq, tile):
            qs = slice(i, i + tile)
            qq = q[qs].float().transpose(0, 1)
            qi = torch.arange(i, i + qq.size(1), device=q.device)

            def scores(j):
                kk = k[j : j + tile].float().index_select(1, kid).transpose(0, 1)
                z = (qq @ kk.transpose(1, 2)) * scale
                kj = torch.arange(j, j + kk.size(1), device=q.device)
                valid = (
                    kj[None, :] <= qi[:, None] + offset
                    if causal
                    else torch.ones(
                        (len(qi), len(kj)), dtype=torch.bool, device=q.device
                    )
                )
                return z.masked_fill(~valid, -torch.inf), valid

            lse = qq.new_full((h, len(qi)), -torch.inf)
            for j in range(0, nk, tile):
                z, _ = scores(j)
                lse = torch.logaddexp(lse, z.logsumexp(-1))
            row_mass = qq.new_zeros((h, len(qi), n_groups))
            row_ent = qq.new_zeros((h, len(qi)))
            for j in range(0, nk, tile):
                ks = slice(j, j + tile)
                z, valid = scores(j)
                lp = z - lse.unsqueeze(-1)
                p = lp.exp()
                row_mass += p @ khot[ks]
                row_ent -= (p * lp.masked_fill(~valid, 0)).sum(-1)
                legal += (qhot[qs].T @ valid.float() @ khot[ks]).long()
                if sha is not None:
                    sha.update(p.cpu().contiguous().numpy().tobytes())
                if calibration or visit is not None:
                    vv = v[ks].float().index_select(1, kid).transpose(0, 1)
                    if calibration:
                        part = (p @ vv).transpose(0, 1)
                        if j == 0:
                            result["mixture"][qs] = part
                        else:
                            result["mixture"][qs] += part
                    if visit is not None:
                        visit(i, j, p, vv)
            error = torch.maximum(error, (row_mass.sum(-1) - 1).abs().max())
            mass += qhot[qs].T @ row_mass
            ent += row_ent @ qhot[qs]
        if not error <= 1e-5:
            raise ValueError(
                "probability normalization failed (or query has no legal keys)"
            )
        den = counts.float().masked_fill(counts == 0, torch.nan)
        result.update(mass=mass / den[:, None], entropy=ent / den, max_row_error=error)
        if sha is not None:
            result["digest"] = sha.hexdigest()
        return result


def _stream(pack, mode):
    indices = pack["_causal_indices" if mode == "causal" else "_full_indices"]
    name = "causal_seq" if mode == "causal" else "full_only_seq"
    if name in pack:
        return pack[name][: len(indices)]
    return pack["packed_sequence"][indices]


def _all(pack, ci, fi):
    if "all_seq" in pack:
        return pack["all_seq"]
    if "packed_sequence" in pack:
        return pack["packed_sequence"]
    c, f = _stream(pack, "causal"), _stream(pack, "full")
    result = c.new_empty((len(ci) + len(fi), *c.shape[1:]))
    result[ci], result[fi] = c, f
    return result


@torch.no_grad()
def observe_two_way(
    arguments, groups, n_groups, layer, *, sentinel=False, tile=64, calibration=False
):
    """Read the bound arguments to two_way_attention, including optional GEN-specific K.

    The release ABI has one causal and one full split per sample. Check against *both*
    the pass offsets and the physical indices, rather than deriving a mask from labels.
    """
    q, k, v = [arguments["packed_" + s + "_states"] for s in ("query", "key", "value")]
    kn = arguments.get("packed_key_states_normalized")
    if kn is None:
        kn = k
    keys = (
        "sample_offsets",
        "_causal_indices",
        "_full_indices",
        "_causal_seq_offsets",
        "_full_only_seq_offsets",
    )
    for pack in (q, k, kn, v):
        if (
            pack.get("is_sharded", False)
            or pack.get("_caption_seq_offsets") is not None
        ):
            raise ValueError(
                "observer needs complete local keys and single-caption splits"
            )
        if any(
            q[key] is not pack[key] and not torch.equal(q[key], pack[key])
            for key in keys
        ):
            raise ValueError("Q/K/V layout mismatch")
    so, ci, fi, co, fo = (q[key] for key in keys)
    starts, cs, fs = (t.tolist() for t in (so, co, fo))
    if len(set(map(len, (starts, cs, fs)))) != 1 or any(
        t[0] != 0 for t in (starts, cs, fs)
    ):
        raise ValueError("unpaired splits")
    length = len(ci) + len(fi)
    if not torch.equal(
        torch.cat((ci, fi)).sort().values, torch.arange(length, device=ci.device)
    ):
        raise ValueError("non-partitioning indices")
    if (
        not length <= len(groups) <= max(length, starts[-1])
        or (groups[length:] != -1).any()
    ):
        raise ValueError("unexpected real/pad boundary")
    ak, av = _all(kn, ci, fi), _all(v, ci, fi)
    qc, kc, vc, qf = (
        _stream(q, "causal"),
        _stream(k, "causal"),
        _stream(v, "causal"),
        _stream(q, "full"),
    )
    result = []
    for b, (start, stop) in enumerate(zip(starts, starts[1:])):
        if (
            start == length and b == len(starts) - 2
        ):  # framework's explicit paired pad segment
            break
        c, f = slice(cs[b], cs[b + 1]), slice(fs[b], fs[b + 1])
        if not (start < stop <= length and cs[b] < cs[b + 1] and fs[b] < fs[b + 1]):
            raise ValueError("invalid real split")
        ordered = torch.arange(start, stop, device=ci.device)
        if not torch.equal(torch.cat((ci[c], fi[f])), ordered):
            raise ValueError("expected same-sample causal prefix followed by full rows")
        g, cg, fg = groups[start:stop], groups[ci[c]], groups[fi[f]]
        if (g == -1).all() and (groups[start:] == -1).all():
            break  # native physical alignment sample, not an observed source window
        if torch.isin(cg, fg).any():
            raise ValueError("text and GEN groups must be disjoint")
        passes = (
            (qc[c], kc[c], vc[c], cg, cg, True),
            (qf[f], ak[start:stop], av[start:stop], fg, g, False),
        )
        if isinstance(sentinel, bool):
            if sentinel and stop - start != length:
                raise ValueError("select physical sentinel indices for packed samples")
            selected = sentinel
        else:
            if len(set(sentinel)) != len(sentinel) or len(sentinel) > 2:
                raise ValueError("at most two distinct sentinels")
            selected = b in sentinel
        map_head = {0: 0, 35: 31}.get(layer) if selected else None
        if map_head is not None and (stop - start > 4096 or map_head >= qc.size(1)):
            raise ValueError("full map must fit without cropping or head substitution")
        matrix = (
            torch.zeros((stop - start, stop - start), dtype=torch.float16, device="cpu")
            if map_head is not None
            else None
        )
        squares = qc.new_zeros((n_groups, n_groups), dtype=torch.float32)
        parts = []
        for qq, kk, vv, qg, kg, causal in passes:
            partial = None

            def accumulate(i, j, p, values):
                # Sum weighted V across all key tiles BEFORE squaring (cancellation matters).
                nonlocal partial
                heads, rows, keys = p.shape
                if j == 0:
                    partial = p.new_zeros((n_groups, heads, rows, values.size(-1)))
                for group in kg[j : j + keys].unique().tolist():
                    take = kg[j : j + keys] == group
                    partial[group] += p[:, :, take] @ values[:, take]
                if j + keys == len(kk):
                    per_q = partial.square().sum((1, 3)).T  # rows, key groups
                    squares.add_(
                        torch.nn.functional.one_hot(qg[i : i + rows], n_groups)
                        .float()
                        .T
                        @ per_q
                    )
                    partial = None
                if matrix is not None:
                    y = i if causal else len(cg) + i
                    matrix[y : y + rows, j : j + keys] = p[map_head].to(
                        device="cpu", dtype=torch.float16
                    )

            parts.append(
                attention_summary(
                    qq,
                    kk,
                    vv,
                    qg,
                    kg,
                    n_groups,
                    causal=causal,
                    tile=tile,
                    calibration=calibration,
                    digest=causal,
                    visit=accumulate,
                )
            )
        text, gen = parts
        is_text = text["counts"] > 0
        out = dict(
            mass=torch.where(is_text[:, None], text["mass"], gen["mass"]),
            entropy=torch.where(is_text, text["entropy"], gen["entropy"]),
            counts=text["counts"] + gen["counts"],
            legal=text["legal"] + gen["legal"],
            max_row_error=torch.maximum(text["max_row_error"], gen["max_row_error"]),
            text_digest=text["digest"],
            text_rows=is_text.nonzero().flatten(),
        )
        den = out["counts"].float() * (qc.size(1) * vc.size(2))
        out["read_strength"] = (
            squares / den.masked_fill(den == 0, torch.nan)[:, None]
        ).sqrt()
        if matrix is not None:
            out["full_map"] = matrix
        if calibration:
            out["mixture"] = torch.cat([part["mixture"] for part in parts])
        result.append(out)
    if not result or any(x < 0 for x in torch.diff(so).tolist()) or starts[-1] < length:
        raise ValueError("invalid sample offsets")
    if not isinstance(sentinel, bool) and not set(sentinel) <= set(range(len(result))):
        raise ValueError("sentinel is absent from the real sample layout")
    return result


class FrozenText:
    """Fixed-prompt integrity index. Snapshot state with the event, not before publish.

    Same reference (e.g. crash/retry) serializes the original payload again; later
    events index it by pointer. Full sentinel matrices are never truncated.
    The key must include the stable prompt, branch, layout, layer and kernel identity.
    """

    def __init__(self, records=None):
        self._records = dict(records or {})

    def state_dict(self):
        return self._records.copy()

    def apply(self, key, sample_summary, reference):
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(reference, str)
            or not reference
        ):
            raise ValueError(
                "fixed text observations need a key and a durable reference"
            )
        result = sample_summary.copy()
        digest = result["text_digest"]
        original, pointer = self._records.get(key, (digest, reference))
        if original != digest:
            raise ValueError(
                "frozen text probabilities changed for an identical prompt binding"
            )
        self._records[key] = (original, pointer)
        result["text_reference"] = pointer
        keep = torch.arange(len(result["counts"]), device=result["counts"].device)
        if pointer != reference:
            keep = keep[~torch.isin(keep, result["text_rows"])]
            result["mass"] = result["mass"][:, keep, :]
            result["entropy"] = result["entropy"][:, keep]
        result["mot_query_groups"] = keep
        return result
