"""Reference engines: the spec's inline v11.0 implementation (parity oracle)."""

from .spec_v11_reference import GASBayesSHAP as SpecReferenceGASBayesSHAP, safe_std as ref_safe_std

__all__ = ["SpecReferenceGASBayesSHAP", "ref_safe_std"]
