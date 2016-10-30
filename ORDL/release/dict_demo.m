function dictAtoms = dict_demo(D)
vNum = 4;
hNum = 8;
patchSize = 12;
atomNum = 32;
vD = ones(3, hNum * (3+patchSize) + 3);
kk = 0;
for jj = 1 : vNum
    hD = ones(patchSize, 3);
    for ii = 1 : hNum
        kk = kk + 1;
        if kk <= atomNum
            Dpatch = reshape(D(:,kk),[patchSize, patchSize]);
        else
            Dpatch = ones(patchSize);
        end
        hD = [hD, (Dpatch - min(Dpatch(:)))/(max(Dpatch(:)) - min(Dpatch(:))), ones(patchSize,3)];
    end 
    vD = [vD; hD ;ones(3, size(hD,2))];
end
dictAtoms = vD;
%figure;imshow(dictAtoms)
end