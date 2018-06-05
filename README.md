# Radio Combine

This repository will hold the scripts and documentation for analyzing combinations of single dish and interferometry radio data

## Scripts

**Compare**
Creates comparison data between two images based on the radial average of their Fourier transform.

```
CASA <1>: run scripts/compare.py -h
usage: compare.py [-h] [-r] [-w binwidth] [-s filename] [--no-plot]
                  image_a image_b

Compare the two images by getting the PSD, binning them, and getting the ratio
of the two. Ideally this raito should be 1.0 near the 0 uv point

positional arguments:
  image_a               path to the first image
  image_b               path to the second image

optional arguments:
  -h, --help            show this help message and exit
  -r, --regrid          regrids image_b to image_a's coordinates
  -w binwidth, --width binwidth
                        The binwidth for binning the PSDs
  -s filename, --save filename
                        Save the final image at the given filename.
  --no-plot             Does not plot the final output. Useful if no X11
                        display
```

Basic example:
```
CASA<1>: run compare.py data/orion.gbt.im data/orion.gbt.noisy.im
```

**Simulate**
This will create simulated data for single dish and intereferometer modes using `simobserve`.

```
CASA <1>: run scripts/simulate.py -h
usage: simulate.py [-h] path

positional arguments:
  path        The path and filename prefix. The files will be saved as
              <path>.sd.im and <path>.int.im

optional arguments:
  -h, --help  show this help message and exit
```

## Tests

In order to run tests, the CASA python needs to be located and unittest must be run. For instance, my CASA python is located at `/home/casa/packages/RHEL6/release/current/bin/python`. 

To run the unittests 
```
$ <CASA python bin> -m unittest discover -v
```

This will run a test suite runner for all scripts. If a specific script is to be tested, use the explicit test script for that
```
$ <CASA python bin> -m unittest -v tests.test_<script>
```
## Acknowledgements

This work was funded by the National Science Foundation in partnership with the National Radio Astronomy Observatory. Thank you to Dr. Kumar Golap and Dr. Tak Tsutsumi for their guidance and assistance. 
