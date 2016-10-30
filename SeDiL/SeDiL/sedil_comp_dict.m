function [ D, SepD ] = sedil_comp_dict( params, varargin)%, deltamsg  )
%SEDIL_COMP_DICT Wrapper for SeDiL to be used in place of ksvd in
% ksvddenoise
% 
% MANDATORY: x, trainnum, blocksize
% 
%  Some fields in PARAMS:
%  --------------------------
%
%    'x' - Noisy signal.
%      The signal to denoise (can be multi-dimensional). Should be of type
%      double, and (for PSNR computations) with values within [0,1] (to
%      specify a different range, see parameter 'maxval' below).
%
%    'blocksize' - Size of block.
%      Indicates the size of the blocks to operate on. Should be either an
%      array of the form [N1 N2 ... Np], where p is the number of
%      dimensions of x, or simply a scalar N, representing the square block
%      [N N ... N]. See parameter 'stepsize' below to specify the amount of
%      overlap between the blocks.
%
%    'dictsize' - Size of dictionary to train.
%      Specifies the number of dictionary atoms to train by K-SVD.
%
%    'psnr' / 'sigma' - Noise power.
%      Specifies the noise power in dB (psnr) or the noise standard
%      deviation (sigma), used to determine the target error for
%      sparse-coding each block. If both fields are present, sigma is used
%      unless the field 'noisemode' is specified (below). When specifying
%      the noise power in psnr, make sure to set the 'maxval' parameter
%      as well (below) if the signal values are not within [0,1].
%
%    'trainnum' - Number of training blocks.
%      Specifies the number of training blocks to extract from the noisy
%      signal for K-SVD training.

X = params.x;
P_sz = params.blocksize;
if ~isfield(params, 'dictsize')
    Nbr_of_atoms = 2*P_sz;
else
    Nbr_of_atoms = params.dictsize;
end

Ntrain = size(X, 3);
train = cell(Ntrain, 1);
for i=1:Ntrain
    train{i} = X(:,:,i);
end

IdxSet = {};
no_sep = 0;

[S] = complete_training_set(train, params.trainnum, P_sz);
n_patches = size(S,2);
Learn_para = init_learning_parameters_dict();

Learn_para.d_sz = P_sz(1);

if no_sep == 1
    P_sz            = prod(P_sz);
    Nbr_of_atoms    = prod(Nbr_of_atoms);
    Learn_para.d_sz = sqrt(P_sz(1));
end
% Normalizing the columns of the inital kernel
randn('seed',0);

for i=1:numel(Nbr_of_atoms)
    D = randn(P_sz(i),Nbr_of_atoms(i));
    [U,SS,V]=svd(D,0);
    D = bsxfun(@minus,D,mean(D));
    Learn_para.D{i}           = bsxfun(@times,D,1./sqrt(sum(D.^2,1)));
end

%%
if ~isfield(params, 'max_iter')
    Learn_para.max_iter = 2000;
else
    Learn_para.max_iter = params.max_iter;
end
if ~isfield(params, 'mu')
    Learn_para.mu       = 1e2;    % Multiplier in log(1+mu*x^2)
else
    Learn_para.mu       = params.mu;
end
if ~isfield(params, 'lambda')
    Learn_para.lambda   = 0.0135;%5e-2;  %1e9;%1e-3;    % Lagrange multiplier
else
    Learn_para.lambda = params.lambda;
end
if ~isfield(params, 'kappa')
    Learn_para.kappa    = 0.0129;%1e-1;    %1e4    % Weighting for Distinctive Terms
else
    Learn_para.kappa = params.kappa;
end
if ~isfield(params, 'q')
    Learn_para.q        = [0,1];
else
    Learn_para.q = params.q;
end

%%
% Displaying results every mod(iter,Learn_para.verbose) iterations
Learn_para.verbose  = 20;
max_out_iter = 10;%3
res = cell(max_out_iter);
tic

lambda_end = 7e-3;%5e4;
shrink1  = (lambda_end/Learn_para.lambda)^(1/(max_out_iter-1));
shrink2  = (1e-2/Learn_para.kappa)^(1/(max_out_iter-1));

S = reshape(S,[P_sz,n_patches]);

Learn_para = learn_separable_dictionary(S, Learn_para);
LP = Learn_para;
LP.X=[];
save(sprintf('DICT'),'-struct','LP');
Learn_para.lambda = Learn_para.lambda*shrink1;
Learn_para.X=[];

SepD = Learn_para.D;
D = kron(SepD{2}, SepD{1});
% D = kron(SepD{1}, SepD{2});

end

