clear all
close all
format shortg


% create example matrix solve
Nsize = 10;
h_A = rand(Nsize);
h_b = rand(Nsize,1);
h_x = h_A\h_b

% transfer data to device
d_A  = gpuArray( h_A );
d_b  = gpuArray( h_b );
d_x  = gpuArray( zeros(Nsize,1) );


GaussSolveptx = parallel.gpu.CUDAKernel('GaussSolve.ptx', 'GaussSolve.cu');
threadsPerBlock = 256;
npixel = 256;
GaussSolveptx.ThreadBlockSize=[threadsPerBlock  1];
blocksPerGrid = (npixel  + threadsPerBlock - 1) / threadsPerBlock;
GaussSolveptx.GridSize=[ceil(blocksPerGrid)  1];

[d_x ] = feval(GaussSolveptx ,Nsize,d_A,d_b,d_x );


mysoln = gather(d_x);

norm(mysoln-h_x)
