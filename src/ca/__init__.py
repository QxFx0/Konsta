"""CA package: certificate management split into focused submodules.

Submodules:
    - :mod:`src.ca.crypto` -- cryptographic primitives (key management,
      certificate generation). No dependencies on :mod:`src.config` or
      :mod:`src.llm_processor` so it can be imported from anywhere in the
      codebase without dragging in the rest of the application.
"""
