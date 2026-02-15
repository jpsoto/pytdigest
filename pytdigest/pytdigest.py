#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Apache License, Version 2.0
# https://www.apache.org/licenses/LICENSE-2.0
#
# Copyright (c) 2022 Tomas Protivinsky, All rights reserved.
# Adaptado para PyCapsules de pytdigest._tdigest

from __future__ import annotations
import numpy as np
from numbers import Number
from typing import Union, List, Optional, Iterable
from enum import Enum

from . import _tdigest


class HandlingInvalid(str, Enum):
    Drop = 'drop'
    Raise = 'raise'


class TDigest:
    """TDigest: estimación aproximada de la distribución empírica de datos en una sola pasada.
    Se pueden combinar múltiples TDigests de fragmentos de datos para obtener una estimación
    global. Se puede calcular CDF e inverse CDF (quantiles) aproximados.
    """

    def __init__(self, compression: int = 100):
        """Inicializa un TDigest vacío con el parámetro de compresión."""
        self.compression = compression
        self._tdigest = _tdigest.create(compression)

    def update(self,
               x: Union[Number, np.ndarray],
               w: Optional[Union[Number, np.ndarray]] = None,
               handling_invalid: HandlingInvalid = HandlingInvalid.Drop):
        """Agrega datos al TDigest.

        Args:
            x: valores a agregar
            w: pesos opcionales
            handling_invalid: cómo tratar valores inválidos ['drop', 'raise']
        """

        """Convierte np.array de tamaño 1 a float."""
        if isinstance(x, np.ndarray) and x.size == 1: x = float(x)
        if isinstance(w, np.ndarray) and w.size == 1: w = float(w)

        if isinstance(x, Number):
            if np.isfinite(x):
                if w is None:
                    w = 1.0
                elif not isinstance(w, Number):
                    raise TypeError("Si x es un número, w también debe serlo.")
                elif not np.isfinite(w):
                    if handling_invalid == HandlingInvalid.Raise:
                        raise ValueError("w es inválido.")
                    else:
                        return
                _tdigest.add(self._tdigest, float(x), float(w))
            elif handling_invalid == HandlingInvalid.Raise:
                raise ValueError("x es inválido.")
        elif isinstance(x, np.ndarray):
            x = x.astype(float)
            if w is None:
                w = np.ones_like(x)
            elif isinstance(w, np.ndarray):
                if w.size != x.size:
                    raise TypeError("w debe tener el mismo tamaño que x.")
                w = w.astype(float)
            else:
                raise TypeError("w es de tipo no soportado.")

            invalid = np.isnan(x) | np.isinf(x) | np.isnan(w) | np.isinf(w) | (w < 0)
            if handling_invalid == HandlingInvalid.Raise and np.any(invalid):
                raise ValueError("x o w contienen valores inválidos.")
            x = x[~invalid]
            w = w[~invalid]

            # TBC. Not needed if using add_batch_TBC
            if not x.flags.c_contiguous: x = x.copy()
            if not w.flags.c_contiguous: w = w.copy()

            _tdigest.add_batch(self._tdigest, x, w)
        else:
            raise TypeError("x debe ser número o ndarray.")

    @staticmethod
    def compute(x: Union[Number, np.ndarray],
                w: Optional[Union[Number, np.ndarray]] = None,
                handling_invalid: HandlingInvalid = HandlingInvalid.Drop,
                compression: int = 100) -> TDigest:
        """Crea un TDigest directamente desde los datos."""
        td = TDigest(compression=compression)
        td.update(x, w, handling_invalid=handling_invalid)
        return td

    def cdf(self, at: Union[Number, List, np.ndarray]) -> Union[float, np.ndarray]:
        """CDF aproximado en los puntos indicados."""
        if isinstance(at, list):
            at = np.array(at)
        if isinstance(at, Number):
            return _tdigest.quantile_of(self._tdigest, at)
        elif isinstance(at, np.ndarray):
            if at.ndim > 1:
                raise ValueError("at no puede ser multidimensional.")
            return np.array(_tdigest.cdf_batch(self._tdigest, at))
        else:
            raise TypeError("at debe ser número, list o ndarray.")

    def inverse_cdf(self, quantile: Union[Number, List, np.ndarray]) -> Union[float, np.ndarray]:
        """Calcula cuantiles aproximados (inverse CDF)."""
        if isinstance(quantile, list):
            quantile = np.array(quantile)
        if isinstance(quantile, Number):
            return _tdigest.value_at(self._tdigest, quantile)
        elif isinstance(quantile, np.ndarray):
            if quantile.ndim > 1:
                raise ValueError("quantile no puede ser multidimensional.")
            return np.array(_tdigest.inverse_cdf_batch(self._tdigest, quantile))
        else:
            raise TypeError("quantile debe ser número, list o ndarray.")

    def __iadd__(self, other):
        if not isinstance(other, TDigest):
            raise TypeError("Solo se pueden sumar TDigests.")
        _tdigest.merge(self._tdigest, other._tdigest)
        return self

    def __add__(self, other):
        result = self.__copy__()
        result += other
        return result

    def __copy__(self):
        new = TDigest(self.compression)
        new += self
        return new

    def get_centroids(self):
        """Devuelve los centroides como un array 2D (mean, weight)."""
        centroids_list = _tdigest.get_centroids(self._tdigest)
        return np.array(centroids_list)

    @staticmethod
    def of_centroids(centroids: np.ndarray, compression: float = 100) -> TDigest:
        """Reconstruye un TDigest desde los centroides."""
        if centroids.ndim != 2 or centroids.shape[1] != 2:
            raise TypeError("Centroids debe ser array 2D con dos columnas (mean, weight).")
        td = TDigest(compression)
        td._tdigest = _tdigest.of_centroids(compression, centroids)
        return td

    def scale_weight(self, factor: float):
        """Escala los pesos por un factor."""
        _tdigest.scale_weight(self._tdigest, factor)

    @property
    def weight(self):
        return _tdigest.total_weight(self._tdigest)

    @property
    def mean(self):
        w = self.weight
        if w == 0:
            return np.nan
        return _tdigest.total_sum(self._tdigest) / w


