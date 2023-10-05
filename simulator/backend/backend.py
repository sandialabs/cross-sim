#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from __future__ import annotations

from types import ModuleType, NoneType


class ComputeBackend:
    """Backend singleton to simplify calls to modules that depend on runtime parameters."""

    _instance: ComputeBackend | None = None
    _backend: ModuleType | None = None
    _use_gpu: bool | None = None

    def __new__(cls, **kwargs):
        """Creates a backend object if one does not exist, or returns the existing instance.

        Args:
            **kwargs: Keyword arguments for the ComputeBackend initialization

        Returns:
            ComputeBackend: Backend singleton
        """
        if cls._instance is None:
            # Initialize a new instance if one does not already exist
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, use_gpu: bool | None = None, gpu_num: int | None = None):
        """Creates an object that directs function calls to the appropriate module determined
        by the simulation parameters.

        Args:
            use_gpu (bool | None, optional): Whether to use a gpu enabled backend.
                If None, selection of module is deferred until a bool value is provided.
                Defaults to None.
            gpu_num (int | None, optional): If use_gpu is true, sets the environment variables
                to specify which gpu to use. Defaults to None.

        Raises:
            TypeError: Raised if given non-bool/None value for use_gpu
            ValueError: Raised if different gpu setting is given after initialization
        """
        if not isinstance(use_gpu, (bool, NoneType)):
            raise TypeError(
                f"use_gpu must be a bool or None, got {type(use_gpu)} instead",
            )
        if self._use_gpu != use_gpu:
            # Don't allow gpu to change
            if (use_gpu is not None) and (self._use_gpu is not None):
                raise ValueError(
                    f"Backend cannot be changed after initialization. Cannot change use_gpu from {self._use_gpu} to {use_gpu}",
                )
            if use_gpu is None:
                pass
            elif use_gpu:
                self._use_gpu = use_gpu
                import cupy as cp

                if gpu_num:
                    cp.cuda.Device(gpu_num).use()
                self._backend = cp
                import cupy as cp
            else:
                self._use_gpu = use_gpu
                import numpy as np

                self._backend = np

    @property
    def backend(self):
        return self._backend

    @property
    def use_gpu(self):
        return self._use_gpu

    def __getattr__(self, name):
        """Returns the value for the attribute of the same name on the backend module.

        Args:
            name (str): Attribute to select

        Raises:
            ValueError: Raised if ComputeBackend was never initialized using 'use_gpu'

        Returns:
            Any: Attribute of the same name from the backend module.
        """
        if self._backend is None:
            raise ValueError(
                "Backend never initialized, ensure the option 'use_gpu' was set appropriately before use.",
            )
        return getattr(self.backend, name)

    def _reset(self):
        """Reset the backend. For debugging use."""
        self._instance = None
        self._backend = None
        self._use_gpu = None
