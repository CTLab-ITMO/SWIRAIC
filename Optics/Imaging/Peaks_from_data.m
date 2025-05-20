function Int_pos = Peaks_from_data(inB_data, dist, hight, N)
%%
    inB_data = str2double(inB_data(1:end-1)); inB_data = inB_data(2:end);
    if max(abs(inB_data(:))) > max(inB_data(:)) 
        inB_data = (-1)*inB_data; 
    end
    for i = 1:length(inB_data)-1
        if diff(inB_data(i:i+1,1)) ~= 0 && abs(inB_data(i))~=0.01
            B_data(i,1) = inB_data(i,1);
        end
    end
    
    [up, lo] = envelope(B_data, 40, 'peak'); B_data_lo = B_data - lo; B_data_lo = B_data_lo-min(B_data_lo); B_data_lo = B_data_lo/max(B_data_lo);
    % figure; plot(B_data_lo);
    
    [B,locks] = findpeaks(B_data_lo,"MinPeakHeight",hight,'MinPeakDistance',dist);
    figure; subplot(2,1,1); plot(B_data); hold on; plot(lo); subplot(2,1,2); findpeaks(B_data_lo,"MinPeakHeight",hight,'MinPeakDistance',dist);
%     figure; plot(diff(locks), 'x')
%%
   k = 0;
    for j = 1:length(B)-1
        diff_locks = locks(j+1) - locks(j); k = k + 1; newB(k) = B(j);        
        if diff_locks >= 70*2-2 && diff_locks <=70*3+2
            newB(k+1) = 0.2; newB(k+2) = 0.2; k = k + 2;
        elseif (diff_locks > 70) && (diff_locks < 2500) 
            newB(k+1) = 0.2; k = k + 1;
        end
        if j ==length(B)-1
            newB(k+1)=B(end);
        end
    end
    newB = newB(17:end);
%%
    if length(newB) > 4096
        newB = newB(1:4096);
    end
    
    if length(newB) == N^2
        Int_pos = newB;
    else
        disp(['ОШИБКА, КОЛ-ВО ПИКОВ = ',num2str(length(newB))])
    end

end
