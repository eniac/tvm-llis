# TVM with LLIS Support

## Build

Follow the instruction of the original TVM. It should be built with CUDA support.


To enable LLIS support, add the following to config.cmake:
```
set(USE_LLIS /path/to/llis)
```
Inside `/path/to/llis`, it should include a directory called `include` which contains the headers of LLIS, and a directory called `lib`, which contains the libraries of LLIS.

When executing a TVM script that targets `cuda-llis`, the environment variable `LLIS_PATH` should be set to `/path/to/llis`.

