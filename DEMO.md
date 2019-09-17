# Demo

## Requirements

- CMake

- Git

- C/C++ compiler
- OpenCV

## Linux

### How to build

```
$ git clone https://github.com/garethwang/ImageMatching.git
$ cd ImageMatching
$ mkdir build
$ cd build
$ cmake ../
$ make
```

### How to run

```
$ cd ..
$ ./build/demo_im
```

## Windows

### How to build

```
$ git clone https://github.com/garethwang/ImageMatching.git
$ cd ImageMatching
$ mkdir build
$ cd build
$ cmake-gui ../
```

- Click  on Configure to process CMakeLists.txt
- Set the OpenCV_DIR to find OpenCV library.
- Click on Configure again.
- Click on Generate.
- Close the cmake-gui.
- Debug

```
$ cd ..
$ cmake --build build
```

- Release

```
$ cd ..
$ cmake --build build --config Release
```

### How to run

- Debug

```
$ ./build/Debug/demo_im.exe
```

- Release

```
$ ./build/Release/demo_im.exe
```

## Result

![](data/result.jpg)

