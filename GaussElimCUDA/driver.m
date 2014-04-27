%function driver

clear all
close all
format shortg

nDim_image = 22;
nDim_matrix = 16;

h_A = randn(nDim_matrix,nDim_matrix,nDim_image,nDim_image);
h_b = randn(nDim_matrix,nDim_image,nDim_image);
h_x = zeros(nDim_matrix,nDim_image,nDim_image);

%for i = 1:nDim_image
%    for j = 1:nDim_image
%        h_A(:,:,i,j) = [6,-1,-2;-6,13,-6;-2,-1,6];
%        h_b(:,i,j) = [3;1;3];
%    end
%end

% transfer data to device
d_A  = gpuArray( h_A );
d_b  = gpuArray( h_b );
d_x  = gpuArray( h_x );

ParallelGaussElimptx = parallel.gpu.CUDAKernel('ParallelGaussElim.ptx', 'ParallelGaussElim.cu');
threadsPerBlock = 256;
npixel = 256;
ParallelGaussElimptx.ThreadBlockSize=[threadsPerBlock  1];
blocksPerGrid = (npixel + threadsPerBlock -1) / threadsPerBlock;
%blocksPerGrid = (npixel  * threadsPerBlock - 1) / threadsPerBlock;
ParallelGaussElimptx.GridSize=[ceil(blocksPerGrid)  1];
%ParallelGaussElimptx.GridSize=[3  1];

[dAout,dbout,dxout] = feval(ParallelGaussElimptx,nDim_image,nDim_matrix,d_A,d_b,d_x);

for i=1:nDim_image
    for j=1:nDim_image
        xtest(:,i,j)=h_A(:,:,i,j)\h_b(:,i,j);
        norm(xtest(:,i,j)-dxout(:,i,j))
    end
end

%exit
