#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
from numbers import Number
from typing import Union, List, Iterable, Optional
from enum import Enum

from . import _tdigest  # módulo C compilado como extensión Python


class HandlingInvalid(str, Enum):
    Drop = 'drop'
    Raise = 'raise'


class TDigest:
    """TDigest: estima la distribución empírica aproximada de un conjunto de datos."""

    def __init__(self, compression: int = 100):
        """Inicializa TDigest vacío con el parámetro de compresión."""
        self.compression = compression
        self._tdigest = _tdigest.td_new(compression)

    def __del__(self):
        _tdigest.td_free(self._tdigest)

    def update(self,
               x: Union[Number, np.ndarray],
               w: Optional[Union[Number, np.ndarray]] = None,
               handling_invalid: HandlingInvalid = HandlingInvalid.Drop):
        """Añade datos al TDigest."""
        x = self._unwrap_if_possible(x)
        w = self._unwrap_if_possible(w)

        if isinstance(x, Number):
            if not np.isfinite(x):
                if handling_invalid == HandlingInvalid.Raise:
                    raise ValueError("x inválido")
                return
            if w is None:
                w = 1.0
            elif not isinstance(w, Number):
                raise TypeError("Si x es un número, w debe ser un número")
            elif not np.isfinite(w):
                if handling_invalid == HandlingInvalid.Raise:
                    raise ValueError("w inválido")
                return
            _tdigest.td_add(self._tdigest, float(x), float(w))
        elif isinstance(x, np.ndarray):
            x = x.astype('float')
            if w is None:
                w = np.ones_like(x)
            elif isinstance(w, np.ndarray):
                if w.size != x.size:
                    raise TypeError("w debe tener el mismo tamaño que x")
                w = w.astype('float')
            else:
                raise TypeError("w tiene tipo no reconocido")
            # eliminar valores inválidos
            invalid = np.isnan(x) | np.isinf(x) | np.isnan(w) | np.isinf(w) | (w < 0)
            if handling_invalid == HandlingInvalid.Raise and np.any(invalid):
                raise ValueError("x o w contiene valores inválidos")
            x = x[~invalid]
            w = w[~invalid]
            _tdigest.td_add_batch(self._tdigest, x.size, x, w)
        else:
            raise TypeError("x tiene tipo no reconocido")

    def cdf(self, at: Union[Number, List, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(at, list):
            at = np.array(at)
        if isinstance(at, Number):
            return _tdigest.td_quantile_of(self._tdigest, at)
        elif isinstance(at, np.ndarray):
            quantiles = np.empty_like(at)
            _tdigest.td_cdf_batch(self._tdigest, at.size, at, quantiles)
            return quantiles
        else:
            raise TypeError("at tiene tipo no reconocido")

    def inverse_cdf(self, quantile: Union[Number, List, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(quantile, list):
            quantile = np.array(quantile)
        if isinstance(quantile, Number):
            return _tdigest.td_value_at(self._tdigest, quantile)
        elif isinstance(quantile, np.ndarray):
            values = np.empty_like(quantile)
            _tdigest.td_inverse_cdf_batch(self._tdigest, quantile.size, quantile, values)
            return values
        else:
            raise TypeError("quantile tiene tipo no reconocido")

    def __iadd__(self, other):
        if not isinstance(other, TDigest):
            raise TypeError("Solo se puede sumar TDigest a TDigest")
        _tdigest.td_merge(self._tdigest, other._tdigest)
        return self

    def __add__(self, other):
        result = self.__copy__()
        result += other
        return result

    def __copy__(self):
        other = TDigest(self.compression)
        other += self
        return other

    def force_merge(self):
        _tdigest.merge(self._tdigest)

    @property
    def weight(self):
        return _tdigest.td_total_weight(self._tdigest)

    @property
    def mean(self):
        if self.weight == 0:
            return np.nan
        return self._tdigest.mean

    @property
    def variance(self):
        if self.weight == 0:
            return np.nan
        return self._tdigest.variance

    @staticmethod
    def _unwrap_if_possible(x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]:
        if isinstance(x, np.ndarray) and x.size == 1:
            x = float(x)
        return x

