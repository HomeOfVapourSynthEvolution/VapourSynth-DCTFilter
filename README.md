Description
===========

For each n x n block (n defaults to 8), DCTFilter will do a Discrete Cosine Transform (DCT), scale down the selected frequency values, and then reverse the process with an Inverse Discrete Cosine Transform (IDCT).

Requires libfftw3f, and on Windows, the release statically links it so there is no need to obtain it separately.


Usage
=====

    dctf.DCTFilter(clip clip, float[] factors[, int[] planes, int n = 8])

* n: the size of the DCT block.

* clip: Clip to process. Any planar format with either integer sample type of 8-16 bit depth or float sample type of 32 bit depth is supported.

* factors: A list of `n` or `n*n` floating point numbers, all of which must be specified as in the range (0.0 <= x <= 1.0).
  - If only `n` numbers are provided, these correspond to scaling factors for the n rows and columns of the 8x8 DCT blocks. The leftmost number corresponds to the top row, left column. This would be the DC component of the transform and should always be left as 1.0. The row & column parameters are multiplied together to get the scale factor for each of the 64 values in a block.
  - If `n*n` numbers are provided, these are used verbatim (row-major).

* planes: A list of the planes to process. By default all planes are processed.


Compilation
===========

Requires `fftw3f`.

```
meson build
ninja -C build
```

or

```
./autogen.sh
./configure
make
```
