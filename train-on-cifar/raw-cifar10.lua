require "torch"
require "torch-env"
require "sys"
require "nn"
require "nndx"

dofile('cifar10.lua')

----------------------------------------------------------------------
----------       YUV, normalized & cached CIFAR10 loader     ---------
----------------------  see caching script!  -------------------------
----------------------------------------------------------------------

nextBatch = getBatchLoader('cifar-10-batches-t7/raw')

