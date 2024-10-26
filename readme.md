# This is the repo for recording my learning steps on massive parallel programming

## Installing CUDA (nightmare for beginner)

I strongly suggest you install it after learning chapter 3. Otherwise you may be very likely to give up.

### Download CUDA toolkit

Get it from [NVIDIA](https://developer.nvidia.com/cuda-downloads) and choose a suitable version of it.
Then set the environment path(**very important!!**)
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\bin`

To varify your installation, use cmd with order:'nvcc --version'
With an information with the version, you have successfully installed it.

### Setup your IDE(VScode)

Don't forget download extension:C/C++.

And set default compiler as cl.exe(in .../MSVC/bin/x64)(i gotta replace it).
