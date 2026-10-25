"""Named seed-derivation schemes used across the package and the experiments.

Every experiment derives its random streams from identifiers (site, draw,
candidate, ...) rather than from a global generator, so a run is reproducible
cell by cell. Historically each experiment wrote its own derivation. This module
collects them under one name each. It does not unify them.

Do not unify these schemes. Each one fixes the draws behind published results.
Switching an experiment to another scheme changes every draw it makes, and the
results would then have to be regenerated.

Schemes and their users:

* :func:`sha256_seed` -- SHA-256 of the ``|``-joined parts, first four bytes
  little-endian. Used by :mod:`causal_grounding.bayesian_modular_naturalness`
  (Experiment III, cut-Bayesian pipeline) and
  :mod:`experiments.exp31_bayesian_manylabs_naturalness`.
* :func:`crc32_seed` -- CRC32 of the ``|``-joined parts. Used by
  :mod:`experiments.naturalness_comparison_data` and
  :mod:`experiments.naturalness_comparison` (Exp28 Twins comparison), and by
  the effect-shift seeds of :mod:`experiments.exp25_tau_floor_regrounding`.
* :func:`blake2b_uniform` -- an 8-byte BLAKE2b digest mapped to ``[0, 1)``.
  Used by :mod:`experiments.exp32_manylabs_noise_candidates` to draw the
  synthetic noise candidates.
* :func:`posterior_sample_seeds` -- children of
  ``SeedSequence([random_seed, POSTERIOR_SAMPLE_SEED_SALT])``. Used by
  :mod:`experiments.osrct_target_pipeline` and by
  :func:`causal_grounding.target_bounds.target_probability_draw_seeds`.
* Integer offsets -- not a function here. ``config.seed + 20_000 +
  target_index`` seeds the per-target grounders of exp16, exp25 and exp32.
  :mod:`experiments.exp25_tau_floor_regrounding` adds
  ``ABLATION_PRIOR_SEED_OFFSET = 41`` for its prior ablation, and
  :mod:`experiments.benchmarks` adds ``+1``, ``+2`` and ``+6`` to separate
  nuisance learners. Those call sites keep their literal offsets.
"""

from __future__ import annotations

import hashlib
from typing import Any
import zlib

import numpy as np

#: Second entropy word of the per-draw posterior-sample seeds (a prime).
POSTERIOR_SAMPLE_SEED_SALT = 104729


def _joined(parts: tuple[Any, ...]) -> bytes:
    """Return the UTF-8 bytes of the parts' text forms joined by ``|``."""
    return "|".join(map(str, parts)).encode("utf-8")


def sha256_seed(*parts: Any) -> int:
    """Derive a reproducible 32-bit seed from the SHA-256 digest of the parts.

    Python randomises string hashing per process, so ``hash`` cannot seed a
    reproducible run. This scheme hashes the parts' text instead.

    Args:
        *parts: Values whose ``str`` forms identify the stream.

    Returns:
        The first four digest bytes read little-endian, in ``[0, 2**32)``.
    """
    return int.from_bytes(hashlib.sha256(_joined(parts)).digest()[:4], "little")


def crc32_seed(*parts: Any) -> int:
    """Derive a reproducible 32-bit seed from the CRC32 checksum of the parts.

    Args:
        *parts: Values whose ``str`` forms identify the stream. One string
            part gives ``zlib.crc32(part.encode())``.

    Returns:
        A seed in ``[0, 2**32)``.
    """
    return zlib.crc32(_joined(parts))


def blake2b_uniform(text: str) -> float:
    """Map a text key to a reproducible uniform draw in ``[0, 1)``.

    Args:
        text: The key. Callers build it themselves, for example
            ``f"{seed}|{index}|{key}"``.

    Returns:
        An 8-byte BLAKE2b digest of ``text``, read little-endian and divided
        by ``2**64``.
    """
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") / float(2**64)


def posterior_sample_seeds(random_seed: int, n_samples: int) -> list[int]:
    """Return one independent seed per posterior sample.

    Args:
        random_seed: Seed of the run.
        n_samples: Number of posterior samples.

    Returns:
        ``n_samples`` seeds. Seed ``k`` is the first 32-bit state word of child
        ``k`` of ``SeedSequence([random_seed, POSTERIOR_SAMPLE_SEED_SALT])``.
    """
    sequence = np.random.SeedSequence([int(random_seed), POSTERIOR_SAMPLE_SEED_SALT])
    return [int(child.generate_state(1)[0]) for child in sequence.spawn(int(n_samples))]
