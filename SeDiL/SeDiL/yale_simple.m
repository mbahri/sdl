%clear;
%restoredefaultpath
%d = pwd;
%cd /vol/bitbucket/mb2215/Thesis/bilinear-rpca
%init
%cd(d)
addpath(genpath('FISTA-master'))
addpath(genpath('ksvdprivate'))
addpath(genpath('utilis'))

% [O, X] = yale_sp(0.25, 0.1);
% x = X(:,:,1);
% xx = x(:,:,1);
% %o = O(:,:,1);

%% Learning

params.x = x;
params.blocksize = [8 8]
%params.blocksize = size(xx);
params.trainnum = 40000;
params.dictsize = 2*params.blocksize;
%params.dictsize = [4*prod(params.blocksize), 8];
params.lambda = 0.1 / prod(params.dictsize);
params.kappa = 0.1 / prod(params.dictsize)
D = sedil_comp_dict(params);

%% Denoising

blocks = im2colstep(xx, [8 8]);
blocks_den = zeros(size(blocks));
opt.verbose = false;
opt.lambda = 1e-1

for i=1:size(blocks, 2)
fprintf('Denoising block %d/%d...\n', i, size(blocks, 2));
blocks_den(:,i) = D*fista_lasso(blocks(:,i), D, [], opt);
end

denoised = col2imstep(blocks_den, size(xx), [8 8]);
%% ddisp(denoised)
% y = D*fista_lasso(reshape(xx, 48*42, 1), D, [], opt);
% denoised = reshape(y, 48, 42);
ddisp(denoised);

% imwrite(denoised, sprintf('%s.png', datetime))
