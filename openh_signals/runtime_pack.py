"""Reconstruct physical labels from a native PackedSequence. No data loader or tokenizer."""

import hashlib
import struct

import torch


GROUPS = (
    "chat",
    "task",
    "caption_metadata",
    "anchor_video",
    "later_video",
    "past",
    "latest",
    "future",
    "memory0",
    "memory1",
    "memory2",
    "other_text",
)
TEXT = {0, 1, 2, 11}


def token_digest(ids):
    """SHA256 of signed 64-bit little-endian physical token ids, including pack specials."""
    return hashlib.sha256(b"".join(struct.pack("<q", int(i)) for i in ids)).hexdigest()


@torch.no_grad()
def label_pack(pack, text_spans, *, memory_rows=0, horizon=24):
    """One camera and one transition payload per logical sample, v1 only.

    text_spans maps the digest of *actual* packed text ids to their disjoint
    semantic classes in order. It must be produced with the real prompt and
    tokenizer offsets, never inferred from a fixed prefix length here.
    """
    if memory_rows not in (0, 3) or horizon != 24:
        raise ValueError("unsupported action topology")
    if any(
        getattr(pack, stream, None) is not None
        for stream in ("sound", "lidar", "radar")
    ):
        raise ValueError("unsupported modality")
    lens = [int(x) for x in pack.sample_lens]
    length = sum(lens)
    if not lens or any(n <= 0 for n in lens) or length != pack.sequence_length:
        raise ValueError("invalid physical sample lengths")
    if list(pack.attn_modes) != ["causal", "full"] * len(lens):
        raise ValueError("expected one causal and one full span per sample")
    ti, vi, ai = [
        v.detach()
        for v in (
            pack.text_indexes,
            pack.vision.sequence_indexes,
            pack.action.sequence_indexes,
        )
    ]
    indices = torch.cat([ti, vi, ai])
    if indices.ndim != 1 or not torch.equal(
        indices.sort().values, torch.arange(length, device=ti.device)
    ):
        raise ValueError("streams must partition every physical row exactly once")
    if len(pack.vision.token_shapes) != len(lens) or len(
        pack.action.token_shapes
    ) != len(lens):
        raise ValueError("cannot bind multisource or missing payload")
    if len(ti) != len(pack.text_ids):
        raise ValueError("text ids and positions disagree")
    groups = torch.full((length,), -1, dtype=torch.int64, device=ti.device)
    offset = 0
    evidence = []
    for sample, n in enumerate(lens):

        def rows(idx):
            return (offset <= idx) & (idx < offset + n)

        selector = rows(ti)
        text, video, action = ti[selector], vi[rows(vi)], ai[rows(ai)]
        for stream in (text, video, action):
            if not len(stream) or not bool((stream.diff() > 0).all()):
                raise ValueError("payload order unknown")
        if tuple(pack.split_lens[2 * sample : 2 * sample + 2]) != (
            len(text),
            len(video) + len(action),
        ):
            raise ValueError("split metadata disagrees with physical positions")
        digest = token_digest(pack.text_ids[selector].tolist())
        labels = text_spans.get(digest)
        if (
            not isinstance(labels, list)
            or len(labels) != len(text)
            or any(type(i) is not int or i not in TEXT for i in labels)
        ):
            raise ValueError("text span/digest not bound to all physical token ids")
        groups[text] = torch.tensor(labels, device=text.device)
        t, h, w = pack.vision.token_shapes[sample]
        if not min(t, h, w) > 0 or len(video) != t * h * w:
            raise ValueError("latent geometry disagrees with vision rows")
        groups[video[: h * w]], groups[video[h * w :]] = 3, 4
        history = len(action) - horizon - memory_rows
        if (
            tuple(pack.action.token_shapes[sample]) != (len(action),)
            or not 0 <= history <= 16
        ):
            raise ValueError("cannot separate history, memory and future")
        for i in range(memory_rows):
            groups[action[i]] = 8 + i
        if history:
            groups[action[memory_rows : memory_rows + history - 1]] = 5
            groups[action[memory_rows + history - 1]] = 6
        groups[action[-horizon:]] = 7
        clean = []
        for modality, expected in ((pack.vision, t), (pack.action, len(action))):
            mask = modality.condition_mask[sample].reshape(-1)
            if mask.numel() != expected or not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError("unrecognized native condition mask")
            clean.append(tuple(mask.bool().tolist()))
        if not all(clean[1][: history + memory_rows]):
            raise ValueError("clean prefix unexpectedly noised")
        evidence.append(
            dict(
                text_digest=digest,
                history=history,
                memory_rows=memory_rows,
                shape=(int(t), int(h), int(w)),
                offset=offset,
                length=n,
                clean_vision=clean[0],
                clean_action=clean[1],
            )
        )
        offset += n
    return groups, evidence
