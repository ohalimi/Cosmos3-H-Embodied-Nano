import pytest

from openh_signals.runtime import FRAMEWORK, S1_PATCH, hook_blob


def test_exact_patch_binding():
    commit = "aeb0f03e8a652d346dcd20b24d88e6eb558b2ba8"
    path = "cosmos3/_src/imaginaire/trainer.py"
    tracked = "774398a5238ab9d79a0c94726b1d72875778644b"
    patched = "5d9cfcf2a7b91799e2999dd151838c2284353bb1"
    assert hook_blob(commit, path, tracked, None) == tracked
    assert hook_blob(commit, path, tracked, S1_PATCH) == patched
    assert hook_blob(commit, "other.py", tracked, S1_PATCH) == tracked
    for pin, base, digest in (
        (FRAMEWORK, tracked, S1_PATCH),
        (commit, "0" * 40, S1_PATCH),
        (commit, tracked, "0" * 64),
    ):
        with pytest.raises(ValueError):
            hook_blob(pin, path, base, digest)
