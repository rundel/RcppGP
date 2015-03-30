BIN               := RcppGP.so

# compilers
NVCC              := /usr/local/cuda/bin/nvcc
#CXX              := g++

# flags
LIBS              :=  -Wl,--export-dynamic -fopenmp -Wl,-rpath,/home/vis/cr173/local/lib/ -Wl,-rpath,/home/vis/cr173/local/lib64/ -L/home/vis/cr173/local/lib64/R/lib -lR -lrt -ldl -lm -licuuc -licui18n  -lopenblas -llapack $(RCPP_LIBS)
LDFLAGS           := -Wl,-rpath,/home/vis/cr173/local/lib/ -Wl,-rpath,/home/vis/cr173/local/lib64/ -lboost_timer -lboost_system
CPPFLAGS          :=  -I/home/vis/cr173/local/lib64/R/include -O3 -I/home/vis/cr173/R/x86_64-unknown-linux-gnu-library/3.1/Rcpp/include -I"/home/vis/cr173/R/x86_64-unknown-linux-gnu-library/3.1/RcppArmadillo/include" -I/usr/local/include -fpic -I../include
NCFLAGS           := -arch=sm_20 -O2 -Xcompiler "-O2 " -I../include

# files
CPP_SOURCES       := $(wildcard *.cpp)
CU_SOURCES        := 
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
