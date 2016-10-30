function D = Mairal_2010_DL(X,scale)
     
    
    param.K=scale;  % learns a dictionary with 100 elements
    param.lambda=0.05;
    param.numThreads=4; % number of threads
    param.batchsize=200;

    param.iter=500;  % let us see what happens after 1000 iterations.
    curPath = pwd;

    cd(fullfile(curPath,'Mairal_DL/spams-matlab'))

    D = mexTrainDL(X,param);
    cd(curPath)
    
end