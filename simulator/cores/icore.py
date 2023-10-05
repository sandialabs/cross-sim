#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

"""Core interface."""

from abc import abstractmethod, ABCMeta


class ICore(metaclass=ABCMeta):
    """Represents the minimum subset of functions that any MVM core must implement (and provide an end-user).

    There are also some convenience functions that were copied over from the (now-defunct) :py:class:`NeuralCore`.
    """

    @property
    def range(self):
        return self.max - self.min

    @abstractmethod
    def set_matrix(self, matrix, applyErrors=True):
        r"""Sets the matrix to use.

        :math:`\mathbf{matrix}\leftarrow\mathtt{matrix}`
        """
        raise NotImplementedError

    @abstractmethod
    def set_vmm_inputs(self, vector):
        r"""Sets the vector to use for :meth:`run_xbar_vmm`.

        :math:`\overrightarrow{vector\_vmm}\leftarrow\mathtt{vector}`
        """
        raise NotImplementedError

    @abstractmethod
    def set_mvm_inputs(self, vector):
        r"""Sets the vector to use for :meth:`run_xbar_mvm`.

        :math:`\overrightarrow{vector\_mvm}\leftarrow\mathtt{vector}`
        """
        raise NotImplementedError

    @abstractmethod
    def run_xbar_vmm(self):
        r"""Returns :math:`\overrightarrow{vector\_vmm}\cdot\mathbf{matrix}`."""
        raise NotImplementedError

    @abstractmethod
    def run_xbar_mvm(self):
        r"""Returns :math:`\mathbf{matrix}\cdot\overrightarrow{vector\_mvm}`."""
        raise NotImplementedError

    @abstractmethod
    def _read_matrix(self):
        """Read the internal matrix held by this core (debug method).

        Data is corrected for scaling at the level it is observed.

        No quantization or other errors are applied.
        """
        raise NotImplementedError

    @abstractmethod
    def _save_matrix(self):
        """Save the internal matrix held by this core (debug method).

        Unlike _read_matrix, all data necessary to restore the matrix is provided.

        No quantization or other errors are applied.
        """
        raise NotImplementedError

    @abstractmethod
    def _restore_matrix(self, matrix):
        """Restore an internal matrix held by this core (debug method).

        You should only use a matrix obtained from _save_matrix, as _read_matrix may remove needed values (e.g.: from an offset core).

        No quantization or other errors are applied.
        """
        raise NotImplementedError
