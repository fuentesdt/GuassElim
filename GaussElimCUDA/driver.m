%function driver

clear all
close all
format shortg


% create example matrix solve
%Nsize = 17;
%h_A = rand(Nsize)+1*eye(Nsize);
%h_A = [6,-1,-2;-6,13,-6;-2,-1,6];
%h_b = rand(Nsize,1);
%cond(h_A)
%h_b = [3;1;3];
%h_C = [h_A,h_b];
%h_C = [6,-1,-2,3;-6,13,-6,1;-2,-1,6,3]
%[L,U]=lu(h_A);
%h_x = U\(L\h_b)
%h_x = h_A\h_b

nDim_image = 3;
nDim_matrix = 3;

%h_A = randn(nDim_matrix,nDim_matrix,nDim_image,nDim_image);
%h_b = randn(nDim_matrix,nDim_image,nDim_image);
h_x = zeros(nDim_matrix,nDim_image,nDim_image);

h_A(:,:,1,1) = [6,-1,-2;-6,13,-6;-2,-1,6];
h_A(:,:,1,2) = [6,-1,-2;-6,13,-6;-2,-1,6];
h_A(:,:,2,1) = [6,-1,-2;-6,13,-6;-2,-1,6];
h_A(:,:,2,2) = [6,-1,-2;-6,13,-6;-2,-1,6];
h_A(:,:,3,1) = [6,-1,-2;-6,13,-6;-2,-1,6];
h_A(:,:,1,3) = [6,-1,-2;-6,13,-6;-2,-1,6];
h_A(:,:,2,3) = [6,-1,-2;-6,13,-6;-2,-1,6];
h_A(:,:,3,2) = [6,-1,-2;-6,13,-6;-2,-1,6];
h_A(:,:,3,3) = [6,-1,-2;-6,13,-6;-2,-1,6];

h_b(:,1,1) = [3;1;3];
h_b(:,1,2) = [3;1;3];
h_b(:,2,1) = [3;1;3];
h_b(:,2,2) = [3;1;3];
h_b(:,2,3) = [3;1;3];
h_b(:,1,3) = [3;1;3];
h_b(:,3,1) = [3;1;3];
h_b(:,3,2) = [3;1;3];
h_b(:,3,3) = [3;1;3];

%h_A(1,1,:,:) = [1,2;3,4];
%h_A(1,2,:,:) = [5,6;7,8];
%h_A(2,1,:,:) = [9,10;11,12];
%h_A(2,2,:,:) = [13,14;15,16];

%h_A

% transfer data to device
d_A  = gpuArray( h_A );
d_b  = gpuArray( h_b );
d_x  = gpuArray( h_x );
%d_C  = gpuArray( [h_A,h_b] );
%d_Pivot  = gpuArray( zeros(Nsize,Nsize+1) );

ParallelGaussElimptx = parallel.gpu.CUDAKernel('ParallelGaussElim.ptx', 'ParallelGaussElim.cu');
threadsPerBlock = 256;
npixel = 256;
ParallelGausElimptx.ThreadBlockSize=[threadsPerBlock  1];
blocksPerGrid = (npixel  + threadsPerBlock - 1) / threadsPerBlock;
ParallelGaussElimptx.GridSize=[ceil(blocksPerGrid)  1];

[dAout,dbout,dxout] = feval(ParallelGaussElimptx,nDim_image,nDim_matrix,d_A,d_b,d_x);
%h_Pivot = gather(d_Pivot);
%h_C = gather(d_C);
%dAout

% Backward substition
%mysoln = dAout(:,Nsize+1);
%for i=Nsize:-1:1
%    for j=Nsize:-1:i+1
%        mysoln(i) = mysoln(i) - dAout(i,j)*mysoln(j);
%    end
%    mysoln(i) = mysoln(i)/dAout(i,i);
%end
%mysoln
%mysoln = gather(d_Pivot);

%norm(mysoln-h_x)

for i=1:nDim_image
    for j=1:nDim_image
        xtest(:,i,j)=h_A(:,:,i,j)\h_b(:,i,j);
        norm(xtest(:,i,j)-dxout(:,i,j))
    end
end

%norm(xtest-dxout)

%exit
