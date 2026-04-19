# DCTFilter

For each 8x8 block, DCTFilter will do a Discrete Cosine Transform (DCT), scale down the selected frequency values, and then reverse the process with an Inverse Discrete Cosine Transform (IDCT).


## Parameters

```py
dctf.DCTFilter(vnode clip, float[] factors[, int[] planes=[0, 1, 2]])
```

- clip: Clip to process. Any format with either integer sample type of 8-16 bit depth or float sample type of 32 bit depth is supported.

- factors: A list of 8 floating point numbers, all of which must be specified as in the range (0.0 <= x <= 1.0). These correspond to scaling factors for the 8 rows and columns of the 8x8 DCT blocks. The leftmost number corresponds to the top row, left column. This would be the DC component of the transform and should always be left as 1.0. The row & column numbers are multiplied together to get the scale factor for each of the 64 values in a block.

- planes: Specifies which planes will be processed. Any unprocessed planes will be simply copied.


## Installation

```
pip install -U vapoursynth-dctfilter
```
