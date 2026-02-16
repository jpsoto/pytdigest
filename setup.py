from setuptools import setup, Extension
import numpy as np


# Definición de la extensión C
tdigest_extension = Extension(
    "pytdigest._tdigest",
    sources=["pytdigest/tdigest_module.c"],
    include_dirs=[np.get_include()],  # <-- Para incluir numpy/arrayobject.h
)

# Configuración del paquete
setup(
    name="pytdigest",
    version="0.1.4",
    description="Python package for *fast* T-Digest calculation.",
    packages=["pytdigest"],
    install_requires=[
        "numpy>=1.19.0",
    ],
    ext_modules=[tdigest_extension],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    author="Tomas Protivinsky",
    author_email="tomas.protivinsky@gmail.com",
    url="https://github.com/protivinsky/pytdigest",
    license="Apache-2.0",
)

