# A Code-base that reproduces the RELIGHT paper
paper: [Relighting humans: occlusion-aware inverse rendering for full-body human images: ACM Transactions on Graphics: Vol 37, No 6](https://dl.acm.org/doi/10.1145/3272127.3275104)

## File Folder Introduction

```shell
D:.
├─app		# Entrance of the project
├─assets	# any assets
├─data      # store datas here
├─lib	   	# Librarys, most from PIFu
│  ├─model		# Load model(.obj files)
│  └─renderer	# render datas
│      └─gl			# pyOpenGL
│         └─data		# Graphics Library Shader Languages
└─util		# Utilities
```

## Issue:
### PRT:
You should install the conda package "pyembree", which is not support for windows using conda.
For windows, plz reference to [this](https://github.com/scopatz/pyembree/issues/14).
OR USE PIP INSTALL AT THE LAST STEP
### PYEXR:
You can reference [this](https://blog.csdn.net/lyw19990827/article/details/123666758).
Or directly:

1. Install openexr first: search at https://www.lfd.uci.edu/~gohlke/pythonlibs/, then download the corresponding version.

2. Then `pip install` it.

3. Then `pip install pyexr`
