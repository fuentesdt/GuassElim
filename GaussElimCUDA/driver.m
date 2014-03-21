clear all
close all
format shortg


% create example matrix solve
Nsize = 10;
h_A = rand(Nsize);
h_b = rand(Nsize,1);
h_C = [h_A,h_b];
h_x = h_A\h_b

% transfer data to device
%d_A  = gpuArray( h_A );
%d_b  = gpuArray( h_b );
d_C  = gpuArray( h_C );
d_x  = gpuArray( zeros(Nsize,1) );


GaussSolveptx = parallel.gpu.CUDAKernel('GaussSolve.ptx', 'GaussSolve.cu');
threadsPerBlock = 256;
npixel = 256;
GaussSolveptx.ThreadBlockSize=[threadsPerBlock  1];
blocksPerGrid = (npixel  + threadsPerBlock - 1) / threadsPerBlock;
GaussSolveptx.GridSize=[ceil(blocksPerGrid)  1];

[d_x ] = feval(GaussSolveptx,Nsize,d_C,d_x);

h_C = gather(d_C);
h_C
%mysoln = gather(d_x);

%norm(mysoln-h_x)
