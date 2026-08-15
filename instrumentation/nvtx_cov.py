import functools
import torch
import torch.cuda.nvtx as nvtx

_EXPECTED_TORCH_PREFIX = "2.9.1"
_installed = False


def _get_param_group_cls():

    candidates = (
        "torch.distributed.fsdp._fully_shard._fsdp_param_group",
        "torch.distributed._composable.fsdp._fsdp_param_group",
    )

    for mod_path in candidates:
        try:
            mod = __import__(mod_path, fromlist=["FSDPParamGroup"])
            return getattr(mod, "FSDPParamGroup")

        except (ImportError, AttributeError):
            continue

    raise AssertionError(
        f"[nvtx_cov] FSDPParamGroup not found — check whether the module path "
        f"changed in torch {torch.__version__}. Paths tried: {candidates}")


def _wrap(cls, method_name: str, tag: str):
    """Replace cls.method_name with an NVTX push/pop sandwich."""
    orig = getattr(cls, method_name)

    assert not hasattr(orig, "_nvtx_cov_orig"), \
        f"[nvtx_cov] {method_name} already patched — duplicate install() call"

    @functools.wraps(orig)
    def wrapped(self, *args, **kwargs):
        nvtx.range_push(self._with_fqn(tag))   # e.g. "AG_unshard (blocks.2)"
        try:
            return orig(self, *args, **kwargs)
        finally:
            nvtx.range_pop()                   # always pop, even on exception

    wrapped._nvtx_cov_orig = orig              # keep the original (dedupe guard + restore)
    setattr(cls, method_name, wrapped)


def install():
    global _installed
    if _installed:
        return

    # fail-fast 1: torch version drift
    ver = torch.__version__.split("+")[0]      # "2.9.1+cu128" → "2.9.1"
    assert ver.startswith(_EXPECTED_TORCH_PREFIX), (
        f"[nvtx_cov] torch version drift: expected {_EXPECTED_TORCH_PREFIX}, "
        f"got {torch.__version__} — hard stop to prevent a repeat of the "
        f"env-drift from the rescue experiment")

    cls = _get_param_group_cls()

    # fail-fast 2: verify the 3 private APIs we depend on exist
    for m in ("unshard", "post_backward", "_with_fqn"):
        assert hasattr(cls, m), (
            f"[nvtx_cov] FSDPParamGroup.{m} missing — FSDP2 internal API "
            f"changed. Re-investigate which functions to patch")

    _wrap(cls, "unshard",       "AG_unshard")
    _wrap(cls, "post_backward", "RS_post_backward")
    _installed = True

    # Runtime evidence — if this line is absent from the log, the patch was NOT applied
    print(f"[nvtx_cov] INSTALLED on {cls.__module__}.{cls.__name__} "
          f"(torch {torch.__version__})", flush=True)