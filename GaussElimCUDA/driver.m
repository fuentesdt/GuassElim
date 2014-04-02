function[] = main()

clear all
close all
format shortg


% create example matrix solve
Nsize = 15;
h_A = rand(Nsize)+1*eye(Nsize);
%h_A = [6,-1,-2;-6,13,-6;-2,-1,6];
h_b = rand(Nsize,1);
%h_b = [3;1;3];
%h_C = [h_A,h_b];
%h_C = [6,-1,-2,3;-6,13,-6,1;-2,-1,6,3]
h_x = h_A\h_b

% transfer data to device
%d_A  = gpuArray( h_A );
%d_b  = gpuArray( h_b );
d_C  = gpuArray( [h_A,h_b] );
d_Pivot  = gpuArray( zeros(Nsize,Nsize+1) );

GaussSolveptx = parallel.gpu.CUDAKernel('GaussSolve.ptx', 'GaussSolve.cu');
threadsPerBlock = 256;
npixel = 256;
GaussSolveptx.ThreadBlockSize=[threadsPerBlock  1];
blocksPerGrid = (npixel  + threadsPerBlock - 1) / threadsPerBlock;
GaussSolveptx.GridSize=[ceil(blocksPerGrid)  1];

[dAout,dPout] = feval(GaussSolveptx,Nsize,d_C,d_Pivot);
h_Pivot = gather(d_Pivot);
h_C = gather(d_C);
%dAout

% Backward substition
mysoln = dAout(:,Nsize+1);
for i=Nsize:-1:1
    for j=Nsize:-1:i+1
        mysoln(i) = mysoln(i) - dAout(i,j)*mysoln(j);
    end
    mysoln(i) = mysoln(i)/dAout(i,i);
end
mysoln
%mysoln = gather(d_Pivot);

norm(mysoln-h_x)

main = 0;

%exit;
