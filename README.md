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

## Building on macOS

Use [Brew](https://brew.sh) to install `FFTW`, `Open MPI`, `ATLAS`, etc. I've tested with FFTW3 and Open MPI; the `configure` script should find everything just fine.

```
% ../configure --with-lapack=apple
```

That should do it.
