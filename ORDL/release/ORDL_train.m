function D = ORDL_train(Data, atomNum, h)
%ORDL_train Online Robust Dictionary Learning
%   D = ORDL_train(Data, atomNum, h) trains dictionary, given a per-defined dictionary atom number and min-batch number.
%   
%   Paras: 
%   @Data      : Input training data - a n x m matrix, where n is the dimemsion
%                of a training sample and m is the number of training samples. 
%   @atomNum   : the number of the dictionary atom. 
%   @h         : the number of training sample in a min-batch
%            
%
%   The Code is created based on the method described in the following paper 
%   [1] "Online Robust Dictionary Learning", Cewu Lu, Jianping Shi, Jiaya Jia, IEEE Conference on Computer Vision and Pattern Recognition, 
%   (CVPR 2013), 2013. 
%   The code and the algorithm are for non-comercial use only.
%  
%   This code is implemented by Cewu Lu (lucewu06@gmail.com)
%   Date  : 04/25/2013
%   Version : 1.0 
%   Copyright 2013, The Chinese University of Hong Kong.
% 

if (~exist('atomNum','var'))
	atomNum = 32;
end

if (~exist('h','var'))
	h = 100;
end 

[dim, samNum] = size(Data); 
lambda = 0.5;
[B,C,D]  = ORDL_init(dim, atomNum ); 
%fprintf('As data growth, the updating speed will be faster! \n');
for ii = 1 : h : samNum
    % L1 data term + L1 sparse term robust sparse coding for a min-batch
    BETAs = robust_CS_batch(D, Data(:, ii: (ii+h-1)), lambda);
    % Robust dictionary update
    [D,B,C] = online_dict_update(D, Data(:, ii: (ii+h-1)), BETAs', B, C, 'Mewton');
    fprintf('%6.2f %% is done! \n', (100*(ii+h)/(samNum+h)));
end

 fprintf('%6.2f %% is done! \n',100);

end

function [B,C,D] = ORDL_init(Dim, atomNum)
    D = rand(Dim, atomNum);
    D = Dictionary_norm(D);
    C = []; B = [];
    for ii = 1 : Dim
        C(ii).m = zeros(atomNum);
        B(ii).m = zeros(atomNum, 1);
    end
end

function D = Dictionary_norm(D)
    for ii = 1 : size(D,2)
        D(:,ii) = D(:,ii)/norm(D(:,ii), 2);
    end
end

function  BETAs = robust_CS_batch(D, batch_data, lambda)

    BETAs = zeros(size(D,2), size(batch_data,2));
    for ii = 1 : size(batch_data, 2) 
        BETAs(:,ii) = robust_sparseCoding(D, batch_data(:,ii), lambda);
    end
end

function  beta = robust_sparseCoding(D,x,lambda)
    [~, m] = size(D);
    I =  lambda * eye(m);
    A = [D;I];
    b = [x(:); zeros(m,1)];
    beta = L1Regression_Reweighted(A,b);
end

function [x,w] = L1Regression_Reweighted(A,b)
    [n,m] = size(A);
    indexX = (1 : n)';
    indexY = (1 : n)';
    val = ones(n,1);
    W = sparse(indexX,indexY,val);
    x = zeros(m,1);
    for jj = 1 : 100
        Last = x;   
        x = (A'*W*A + 10^(-9)*eye(m))\(A'*W*b);
        w = 1./sqrt(abs(A*x - b).^2 +0.000001  );
        W = diag(w);
        
        if norm(Last - x)/sqrt(norm(Last)*norm(x)) < 0.002
            break;
        end
        
    end
end

function  [D,Bt,Ct] = online_dict_update(D, batch_data, BETAs, B, C, IterMethod)

tol = 1e-6;
MaxIter = 200;
Iter = 0; dif = inf;
W = eye(size(BETAs,1));

    while tol < dif 
        DLast = D ;
        Iter = Iter + 1;
        if Iter == MaxIter
            break;
        end
        Bt = B;
        Ct = C;
        for ii = 1 : size(batch_data,1)
            %if Iter  ~= 1 
            W =  reweighted(batch_data(ii,:)',D(ii,:)', BETAs);
            %end
            Ct(ii).m =  Ct(ii).m  +  BETAs' * W * BETAs;
            Bt(ii).m =  Bt(ii).m  +  BETAs' * (W*batch_data(ii,:)');
            if strcmp('matrix_inversion', IterMethod)
                D(ii,:) = ( ( Ct(ii).m + (1e-8)*eye(size(Ct(ii).m))) \ Bt(ii).m)';
            end
            if strcmp('Mewton', IterMethod)
                D(ii,:) =    Newton_method(Ct(ii).m, Bt(ii).m, D(ii,:)');
            end
        end
        dif = mean(abs(DLast(:) - D(:) ));
    end 
    
end

function Dline = Newton_method(A,b,Init)
    Dline = Init;
    tol = 1e-4;
    MaxIter = 30;
    Iter = 0; dif = inf;
    while tol < dif 
        DLast = Dline ;
        Iter = Iter + 1;
        if Iter == MaxIter
            break;
        end
        d = A*Dline - b;
        a = -d'*A*d/(d'* A'*A*d);
        Dline =  Dline + a*(A*Dline - b);
        dif = mean(abs(DLast(:) - Dline(:) ));
    end
    
    Dline = Dline';
end

function W =  reweighted(sLine,DLine, BETA)
    deta = 0.0000001;
    r = abs(sLine -  BETA*DLine );
    W = eye(length(r));
    for ii = 1 : length(r)
        W(ii,ii) = 1/sqrt(abs(r(ii))^2 + deta);
    end
end


















