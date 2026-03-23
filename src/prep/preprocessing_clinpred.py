# src/prep/preprocessing_clinpred.py - Professional Legacy Wrapper
# This file is maintained for backward compatibility with existing training notebooks.
# ALL core logic has been moved to the unified src/prep/pipeline.py.

from src.prep.pipeline import RegistryImputer, preprocessor, identity_transform

__all__ = ['RegistryImputer', 'preprocessor', 'identity_transform']
