
# cuda
BASE_INCLUDES = -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\include"
BASE_LIBDIRS = -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\lib\x64"

NPPLIBS = -lnppc -lnppicc -lnppist

# add opencv
OPENCV_ROOT = C:\opencv\build
OPENCV_INCLUDES = -I"$(OPENCV_ROOT)\include"
OPENCV_LIBDIR = -L"$(OPENCV_ROOT)\x64\vc16\lib"
OPENCV_LIBS = -lopencv_world4120


EXE := assignment.exe

INPUT_DIR := inputs
INPUT_FILES := $(wildcard $(INPUT_DIR)/*)

# combine
INCLUDES = $(BASE_INCLUDES) $(OPENCV_INCLUDES)
LIBDIRS = $(BASE_LIBDIRS) $(OPENCV_LIBDIR)
LIBS = $(NPPLIBS) $(OPENCV_LIBS)


all: assignment.cu
	nvcc -std=c++17 $(INCLUDES) assignment.cu -o $(EXE) $(LIBDIRS) $(LIBS)


run_all:
	@echo Running $(EXE) on each file...
	@for %%f in ($(INPUT_FILES)) do ( \
		echo Processing %%f... && \
		$(EXE) "%%f" \
	)
