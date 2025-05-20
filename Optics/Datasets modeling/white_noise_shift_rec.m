function G_H = white_noise_shift_rec(size, sk, Ob, RN)
    sumBI = zeros(size); sumI = zeros(size); sumB = 0;
%     Bw_j = zeros(1,RN); I_j = zeros(RN,size,size); I_vec_j = zeros(RN,size^2);
% %     hwb=waitbar(0,['N = ',num2str(RN),' - Calculating...'], 'Name', 'Time marching');
    for j = 1:RN
        S = randi([0 1], [fix(2*size/sk)+1 fix(2*size/sk)+1]); S = kron(S, ones(sk)); 
        x = randi([1 size]); y = randi([1 size]); I = S(x:x+size-1, y:y+size-1);
        B = sum(sum(I.*Ob)); 
        sumBI = sumBI + B.* I; sumI = sumI + I; sumB = sumB + B;
%         Bw_j(j) = B; I_j(j,:,:) = I; I_vec_j(j,:) = reshape(I,1,[]);
% %     waitbar(j/RN, hwb);
    end
% %     close(hwb);
    G_H = sumBI./RN - (sumB/ RN).* (sumI./ RN); G_H = G_H - min(G_H(:)); G_H = G_H/max(G_H(:)); 
%     CS_vec2 =  lsqminnorm(I_vec_j,Bw_j');
%     G_H = reshape(CS_vec2,size,size);
%     figure; 
%     tiledlayout(1,2); nexttile; imshow(Ob); nexttile; 
%     imshow(G_H); 
        
    
    

end