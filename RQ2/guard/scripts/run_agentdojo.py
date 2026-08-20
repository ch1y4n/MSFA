"""agentdojo benchmark entry point with Guard defenses pre-registered.

Run with ``python -m guard.scripts.run_agentdojo <agentdojo CLI args>``.
Importing :mod:`guard.defenses.melon` and :mod:`guard.defenses.promptarmor`
before agentdojo's CLI lets the ``--defense`` click choice include them.
"""

import guard.defenses.melon  # noqa: F401
import guard.defenses.promptarmor  # noqa: F401
import guard.defenses.attriguard  # noqa: F401
from agentdojo.scripts.benchmark import main
import logging

from rich.logging import RichHandler


if __name__ == "__main__":
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
        handlers=[RichHandler(show_path=False, markup=True)],
    )
    main()
