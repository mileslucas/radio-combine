# Radio Combine

This repository will hold the scripts and documentation for analyzing combinations of single dish and interferometry radio data

## Scripts

**Compare**
Creates comparison data between two images based on the radial average of their Fourier transform.

```
<1>: run compare.py -h
usage: compare.py [-h] [-r] [-n] image_a image_b

positional arguments:
  image_a        path to the first image
  image_b        path to the second image

optional arguments:
  -h, --help     show this help message and exit
  -r, --regrid   regrids image_b to image_a's coordinates
  -n, --no-plot
```

```
<1>: run compare.py data/orion.gbt.im data/orion.gbt.noisy.im
```

## Acknowledgements

This work was funded by the National Science Foundation in partnership with the National Radio Astronomy Observatory. Thank you to Dr. Kumar Golap and Dr. Tak Tsutsumi for their guidance and assistance. 
