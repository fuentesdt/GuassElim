#
# file : Makefile (UNIX)
#
#  mexGPUArray   from 
# 
#     $(MATLABROOT)/toolbox/distcomp/gpu/extern/src/mex/mexGPUExample.cu
#
# You can invoke this Makefile using 
#  make -f Makefile MATLABROOT=[directory where MATLAB is installed]
#
# If you do not want to specify the MATLABROOT at the gmake command line 
# every single time, you can define it by uncommenting the line below
# and assigning the correct root of MATLAB (with no trailing '/') on
# the right hand side.
#
MATLABROOT	:= /opt/apps/matlab/2013a/
#

#
# Defaults
#

MEX=$(MATLABROOT)/bin/mex
CUDADIR = /opt/apps/cuda/5.0/

# The following are the definitions for each target individually.

GaussSolve.ptx:   GaussSolve.cu
	$(CUDADIR)/bin/nvcc -g -G -ptx -gencode=arch=compute_20,code=sm_20   GaussSolve.cu


PronyKernel.mexa64: PronyKernel.cu
	$(MEX) -g -v -f glnxa64/mexopts.sh $^

tags:
	ctags -R  --langmap=c++:+.cu $(MATLABROOT) . 
