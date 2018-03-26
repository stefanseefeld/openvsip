# Welcome to OpenVSIP

[![Build Status](https://travis-ci.org/openvsip/openvsip.svg)](https://travis-ci.org/openvsip/openvsip)

OpenVSIP is a high-quality implementation of the 
[VSIPL++ standard](http://portals.omg.org/hpec/content/specifications), originally
developed by CodeSourcery as Sourcery VSIPL++.

This library is free software. For license information please refer to the file [LICENSE](LICENSE).

For a quick overview of the project please refer to the [Getting Started](doc/getting-started.md) document.

# Quick Building Instructions

1. Generate configure script with `./autogen.sh` (only needed once)
2. Create object directory: `mkdir objdir`
3. Make object directory current: `cd objdir`
4. Configure build: `../configure [options]`
5. Build: `make`
6. Test (optional - requires qmtest): `make check`
7. Install: `make install`

By default, the above will compile the source files with `-g -O2`. If by chance you want a pure debug build:

```
CFLAGS="-g" CXXFLAGS="-g" ../configure [options]
```

## Building on macOS

Use [Brew](https://brew.sh) to install [FFTW](http://www.fftw.org), [Open MPI](https://www.open-mpi.org), [OpenBLAS](https://www.openblas.net), [ATLAS](http://math-atlas.sourceforge.net), etc. I've tested with FFTW and Open MPI without issues. I think the `configure` script should find everything on its own in `/usr/local` but if not checkout `configure --help` for options to point to the right location.

To have the software use the LAPACK library found in [Apple's Accelerate framework](https://developer.apple.com/documentation/accelerate), just use:

```
% ../configure --with-lapack=apple
```

