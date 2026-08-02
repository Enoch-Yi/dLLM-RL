#!/usr/bin/env python3
"""Compatibility entry point for the DCoLT communication-safe learner.

The timeout must be installed inside ``rl_sdar_multitrace.main`` at its first
Accelerator construction.  The former monkeypatch happened too late and jobs
18454571/18454572 still used the 600-second default.  Keep this filename for
old launch manifests, but delegate without patching runtime state.
"""

import sys


def main() -> None:
    if "training.registered_method=dcolt" not in sys.argv[1:]:
        raise ValueError("the communication-safe launcher is DCoLT-only")

    import rl_sdar_multitrace as learner
    learner.main()


if __name__ == "__main__":
    main()
