[O, X] = yale_sp(0.25, 0.1);
O = O(:,:,1:64);
X = X(:,:,1:64);

Ntrain = 64;
train = cell(Ntrain, 1);
for i=1:Ntrain
    train{i} = X(:,:,i);
end

P_sz            = [8,8];
Nbr_of_atoms    = round(sqrt(4)*P_sz);

IdxSet              = {};
no_sep              = 0;

n_patches    = 30000*1; % 40k in paper?
[S]   =  complete_training_set(train, n_patches, P_sz);

n_patches    = size(S,2);
Learn_para          = init_learning_parameters_dict();
if exist('LLGG','var')
    Learn_para.logger = LLGG;
end

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

Learn_para.max_iter = 2000;
Learn_para.mu       = 1e2;    % Multiplier in log(1+mu*x^2)
Learn_para.lambda   = 0.0135;%5e-2;  %1e9;%1e-3;    % Lagrange multiplier
Learn_para.kappa    = 0.0129;%1e-1;    %1e4    % Weighting for Distinctive Terms
Learn_para.q        = [0,1];

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

toc
LLGG = Learn_para.logger;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Inference

D = kron(Learn_para.D{2}, Learn_para.D{1});
% vS = reshape(X(:,:,1), 48*42, 1);
vS = reshape(S(:,:,1), 64, 1);
opts.verbose = true;
opts.lambda = 1e-2;
vX = fista_lasso(vS, D, [], opts);

ddisp(reshape(D*vX, 8, 8))