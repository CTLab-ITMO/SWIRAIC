function w = Cal_Sal_Transform_Optimized(n)
    % Cal-Sal transform with optimizations
    w = zeros(2^n, 2^n);
    
    % Precompute the R matrix
    R = zeros(n, n);
    if n == 1
        R = [1];
    else
        for k = 1:n-1
            R(k, n-k+1) = 1;
            R(n-k+1, k+1) = 1;
        end
        R(n, 1) = 1;
    end
    
    % Precompute binary representations of indices
    bin_vecs = zeros(2^n, n);
    for j = 1:2^n
        bin_vecs(j, :) = flip(de2bi(j-1, n));
    end

    % Fill the matrix w, leveraging symmetry and precomputed values
    for j = 1:2^n
        bj = bin_vecs(j, :);  % Precomputed binary vector for j
        for k = j:2^n
            bk = bin_vecs(k, :);  % Precomputed binary vector for k
            entry = (-1)^sum(bj * R * bk');
            w(j, k) = entry;
            w(k, j) = entry;  % Use symmetry
        end
    end
end
