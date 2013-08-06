#
#

BIN               := RcppGP.so

# compilers
NVCC              := /usr/local/cuda/bin/nvcc
#CXX              := g++

# flags
LIBS              :=  -L/usr/lib/R/lib -lR -L/home/rundel/R/library/Rcpp/lib -lRcpp -Wl,-rpath,/home/rundel/R/library/Rcpp/lib -lblas -llapack  -lcudart -lcublas -lmagma $(RCPP_LIBS)
LDFLAGS           :=   -lz -lboost_timer -L/usr/local/cuda/lib64 -L/usr/local/magma/lib
CPPFLAGS          :=  -I/usr/share/R/include -O2 -pipe -g -I/home/rundel/R/library/Rcpp/include -I"/home/rundel/R/library/RcppArmadillo/include"  -I/usr/local/cuda/include -DHAVE_CUBLAS -DUSE_GPU -I/usr/local/magma/include -fpic -I../include
NCFLAGS           := -arch=sm_20 -O2 -Xcompiler "-O2 -fPIC" -I../include

# files
CPP_SOURCES       := $(wildcard *.cpp)
CU_SOURCES        := $(wildcard *.cu)
CPP_OBJS          := $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS           := $(patsubst %.cu, %.o, $(CU_SOURCES))

$(BIN): $(CU_OBJS) $(CPP_OBJS)
		$(CXX) -shared -o $@ $(CU_OBJS) $(CPP_OBJS) $(LDFLAGS) $(LIBS)

%.o: %.cpp
		$(CXX) $(CPPFLAGS) -c $< -o $@

%.o: %.cu
		$(NVCC) $(NCFLAGS) -c $< -o $@

all: $(BIN) 

clean:
		rm -f *.o
