AC_INIT([RcppGP], [0.1.0], [rundel@gmail.com])

#################### Compiler ####################

AC_LANG(C++)
AC_REQUIRE_CPP
AC_PROG_CXX


#################### Boost Stuff ####################

AC_CHECK_LIB(boost_timer-mt, main, [],
    [AC_CHECK_LIB(boost_timer, main, [],
        AC_MSG_FAILURE(["Boost timer library not found."]))])   

AC_CHECK_HEADER([boost/timer.hpp], [], 
    [AC_MSG_ERROR([Missing header: boost/timer.hpp])])

AC_CHECK_HEADER([boost/assign/list_of.hpp], [], 
    [AC_MSG_ERROR([Missing header: boost/assign/list_of.hpp])])


#################### Defaults     ####################

USE_GPU=yes
PREV_CXXFLAGS=$CXXFLAGS 
PREV_CPPFLAGS=$CPPFLAGS 
PREV_LIBS=$LIBS
PREV_LDFLAGS=$LDFLAGS

#################### CUDA Toolkit ####################

AC_ARG_WITH(
    [cuda-home],
    [AC_HELP_STRING([--with-cuda-home=DIR],[full path to CUDA installation @<:@/usr/local# /cuda@:>@])],
    [CUDA_HOME="$withval"],
    [CUDA_HOME=/usr/local/cuda]
)
AC_MSG_RESULT([Setting CUDA home directory to ${CUDA_HOME}])


AC_CHECK_FILE(
    [${CUDA_HOME}/lib64/libcudart.so], 
    [LDFLAGS+=" -L${CUDA_HOME}/lib64"],
    [LDFLAGS+=" -L${CUDA_HOME}/lib"]
)

NVCC=${CUDA_HOME}/bin/nvcc
AC_CHECK_FILE([${NVCC}], [], [USE_GPU=no])

AC_ARG_WITH([arch],
    [AC_HELP_STRING([--with-arch=SM],[compute capability of the target device @<:@sm_30@:>@])],
    [ARCH="$withval"],
    [ARCH=sm_30]
)
AC_MSG_RESULT([Setting CUDA compute capability to ${ARCH}])

R_CPICFLAGS=`"${R_HOME}/bin/R" CMD config CPICFLAGS`
NVCC_CFLAGS="-arch=${ARCH} -O3 -Xcompiler \"${R_CPICFLAGS}\""


CXXFLAGS+=" -fopenmp"

AC_CHECK_LIB([cudart],   [main], [], [USE_GPU=no])
AC_CHECK_LIB([cublas],   [main], [], [USE_GPU=no])
AC_CHECK_LIB([curand],   [main], [], [USE_GPU=no])
AC_CHECK_LIB([cusolver], [main], [], [USE_GPU=no])

CPPFLAGS+=" -I${CUDA_HOME}/include"
AC_CHECK_HEADER([cuda.h], [], [USE_GPU=no])


#################### MAGMA Libraries ####################

AC_ARG_WITH([magma-home],
    [AC_HELP_STRING([--with-magma-home=DIR],[full path to MAGMA installion directory @<:@/usr# /local/magma@:>@])],
    [MAGMA_HOME="$withval"],
    [MAGMA_HOME=/usr/local/magma]
)
AC_MSG_RESULT([Setting Magma home directory to ${MAGMA_HOME}])

LDFLAGS+=" -L${MAGMA_HOME}/lib"

AC_CHECK_LIB([magma], [main], [], [USE_GPU=no])
#AC_CHECK_FILE([${MAGMA_HOME}/include/magma.h], [], [USE_GPU=no])

CPPFLAGS+=" -I${MAGMA_HOME}/include"
AC_CHECK_HEADER([magma.h], [], [USE_GPU=no])


################# Use GPU? #################

if test "${USE_GPU}" == "yes"; then
    CPPFLAGS+=" -DUSE_GPU"
    CU_FILES="\$(wildcard *.cu)"
else
    AC_MSG_RESULT([Unable to locate magma library / headers, compiling without gpu support])
    CXXFLAGS = $PREV_CXXFLAGS 
    CPPFLAGS = $PREV_CPPFLAGS 
    LIBS     = $PREV_LIBS
    LDFLAGS  = $PREV_LDFLAGS
    CU_FILES = ""
fi

# 
# CPPFLAGS+=" -fPIC"
# 
# #AC_MSG_RESULT([CPPFLAGS ${CPPFLAGS}])
# #AC_MSG_RESULT([LIBS ${LIBS}])
# #AC_MSG_RESULT([LDFLAGS ${LDFLAGS}])
# 
# 

#################### Makevars ####################

AC_SUBST([CPPFLAGS])
AC_SUBST([CXXFLAGS])

AC_SUBST([LDFLAGS])
AC_SUBST([LIBS])

AC_SUBST([CU_FILES])

AC_SUBST([NVCC])
AC_SUBST([NVCC_CFLAGS])

AC_CONFIG_FILES([src/Makefile])
AC_OUTPUT
