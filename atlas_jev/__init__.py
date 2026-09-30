"""ATLAS-driven adversarial evaluation of open LLMs with Jev as structured evaluator.

Two things are under measurement in this project, and the code keeps them apart:

* the **target model**, which may or may not succumb to an adversarial prompt, and
* the **evaluator** (Jev), which may or may not correctly report what happened.

Neither is treated as ground truth. See README.md.
"""

__version__ = "0.1.0"
