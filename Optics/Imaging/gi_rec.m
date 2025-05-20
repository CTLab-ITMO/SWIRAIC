clear; clc; 
close all;

dist = 40; hight = 0.27; N = 64;

userInput = input('Введите количество картинок (1 или 3): ', 's');
if strcmpi(userInput, '1')
    name800 = uigetfile('*.*'); inB_data800 = readlines(name800); Int_pos800 = Peaks_from_data(inB_data800,dist, hight, N);
    Int_pos1050 = Int_pos800; Int_pos1550 = Int_pos1050;
elseif strcmpi(userInput, '3')
    name800 = uigetfile('*.*'); inB_data800 = readlines(name800); Int_pos800 = Peaks_from_data(inB_data800,dist, hight, N);
    name1050 = uigetfile('*.*'); inB_data1050 = readlines(name1050); Int_pos1050 = Peaks_from_data(inB_data1050,dist, hight, N);
    name1550 = uigetfile('*.*'); inB_data1550 = readlines(name1550); Int_pos1550 = Peaks_from_data(inB_data1550,dist, hight, N);
else
    disp('Ошибка: Введите 1 или 3.');
end

if N == 128
    type = 3;
elseif N == 64
    type = 1;
end
%%
if type ==1
    H_size = 64;
    G800 = vosst_pos(Int_pos800,H_size); G800 = G800(2:end-1,2:end-1); G800 = G800-min(min(G800)); G800 = G800/max(max(G800));
    GI800 = medfilt2(G800,[5 5]); 
    G1050 = vosst_pos(Int_pos1050,H_size); G1050 = G1050(2:end-1,2:end-1); G1050 = G1050-min(min(G1050)); G1050 = G1050/max(max(G1050));
    GI1050 = medfilt2(G1050,[5 5]); 
    G1550 = vosst_pos(Int_pos1550,H_size); G1550 = G1550(2:end-1,2:end-1); G1550 = G1550-min(min(G1550)); G1550 = G1550/max(max(G1550));
    GI1550 = medfilt2(G1550,[5 5]);
end
%%
if type == 3
    H_size = 128;
    h = hadamard(H_size*H_size);
    h = double(h>0);
    [G800, GI800] = correlation(h,H_size,Int_pos800);
    [G1050, GI1050] = correlation(h,H_size,Int_pos1050);
    [G1550, GI1550] = correlation(h,H_size,Int_pos1550);
end
%% 
if type ==2
    size_x = 64;
    size_y = 64;
    res_i = [num2str(size_x),'_',num2str(size_y)];
    load(['C:\Users\QPM_Lab\Downloads\DMDNetWork\DMD_Matlab\binrand_',res_i,'.mat']);

    [N,size_x_y]=size(a3);
    sBI(1:size_x_y)=0;
    sI = sBI;
    sB = 0;
    N_new = N;
    b_h = Int_pos;
    for i = 1:N_new
        sBI = sBI + b_h(i) .* squeeze(a3(i,:));
        sI = sI + squeeze(a3(i,:));
        sB = sB + b_h(i);
    end

    G_s = sBI ./ N_new - (sB / N_new) .* (sI ./ N_new);
    G = reshape(G_s,[size_x,size_y]);
    GI = medfilt2(G,[5 5]);
    figure
    imshow(imrotate(flip(G),45),[]);
    % colormap jet
    figure
    imshow(imrotate(flip(GI),45+180),[]);
    colormap jet
end
%%

angle = 135;
figure; subplot(2,3,1); imshow(imrotate(G800, angle),[]); title('Image 800 nm')
subplot(2,3,4); imshow(imrotate(GI800,angle),[0.1 0.5]); title('Filtered image 800 nm')
subplot(2,3,2); imshow(imrotate(G1050,angle),[]); title('Image 1050 nm')
subplot(2,3,5); imshow(imrotate(GI1050,angle),[0 0.7]); title('Filtered image 1050 nm')
subplot(2,3,3); imshow(imrotate(G1550,angle),[]); title('Image 1550 nm')
subplot(2,3,6); imshow(imrotate(GI1550,angle),[0 0.7]); title('Filtered image 1550 nm')
% 
G800(44:46, 11:13) = 0.25; G1050(44:46, 11:13) = 0.3; G1550(44:46, 11:13) = 0.3;
G800 = G800-min(min(G800)); G800 = G800/max(max(G800));
G1050 = G1050-min(min(G1050)); G1050 = G1050/max(max(G1050));
G1550 = G1550-min(min(G1550)); G1550 = G1550/max(max(G1550));
    

userInput = input('Записать картинки? Введите да или нет: ', 's');
if strcmpi(userInput, 'да')
    imwrite(G800,['Vlad\800\',num2str(name800),'.jpg']);
    imwrite(G1050,['Vlad\1050\',num2str(name1050),'.jpg']);
    imwrite(G1550,['Vlad\1550\',num2str(name1050),'.jpg']);
elseif strcmpi(userInput, 'нет')
else
    disp('Ошибка: введите "да" или "нет".');
end


%%

function z = vosst(Int_pos,Int_neg)

N = 64;
H = Cal_Sal_Transform_Optimized(2*log2(N));

W = zeros(N, N); % Исходная матрица с числами 1:4096

center =(N / 2); % Центр матрицы

% Направления движения: вправо, вниз, влево, вверх
directions = [0 1; 1 0; 0 -1; -1 0];

% Начальные параметры
row = center;
col = center;
step_size = 1; % Длина перемещения в одном направлении
dir_index = 1; % Индекс направления (вправо)
count = 1;

while count <= N

    for s = 1:2 % Два раза увеличиваем длину шага (один виток спирали)
        for step = 1:step_size
            if row >= 1 && row <= N && col >= 1 && col <= N

                W(row, col) = (Int_neg(count)-Int_pos(count));

                count = count + 1;
            end
            row = row + directions(dir_index, 1);
            col = col + directions(dir_index, 2);
        end
        dir_index = mod(dir_index, 4) + 1; % Меняем направление
    end
    step_size = step_size + 1; % Увеличиваем длину движения
end

W2 = fftshift(W);
W2 = reshape(W2,[N*N,1]);
H_inv = inv(H);
z = H_inv*W2;
z = reshape(z,[N, N]);

end

function z = vosst_pos(Int_pos,H_size)

N = H_size;
H = Cal_Sal_Transform_Optimized(2*log2(N));

W = zeros(N, N); % Исходная матрица с числами 1:4096

center =(N / 2); % Центр матрицы

% Направления движения: вправо, вниз, влево, вверх
directions = [0 1; 1 0; 0 -1; -1 0];

% Начальные параметры
row = center;
col = center;
step_size = 1; % Длина перемещения в одном направлении
dir_index = 1; % Индекс направления (вправо)
count = 1;

while count <= N^2

    for s = 1:2 % Два раза увеличиваем длину шага (один виток спирали)
        for step = 1:step_size
            if row >= 1 && row <= N && col >= 1 && col <= N

                W(row, col) = (Int_pos(count));

                count = count + 1;
            end
            row = row + directions(dir_index, 1);
            col = col + directions(dir_index, 2);
        end
        dir_index = mod(dir_index, 4) + 1; % Меняем направление
    end
    step_size = step_size + 1; % Увеличиваем длину движения
end

W2 = fftshift(W);
W2 = reshape(W2,[N*N,1]);
H_inv = inv(H);
z = H_inv*W2;
z = reshape(z,[N, N]);

end

function [G128, GI128] = correlation(h,H_size,b_h)
    sBI(1:H_size*H_size)=0;
    sI = sBI;
    sB = 0;
    N_new = H_size*H_size;
    for i = 1:N_new
        sBI = sBI + b_h(i) .* squeeze(h(i,:));
        sI = sI + squeeze(h(i,:));
        sB = sB + b_h(i);
    end

    G128 = sBI ./ N_new - (sB / N_new) .* (sI ./ N_new);
    G128 = reshape(G128,[H_size,H_size]);
    G128 = G128(2:end-1,2:end-1); G128 = G128- min(min(G128)); G128 = G128/max(max(G128)); %G = imresize(G, [length(range_x),length(range_y)]);
    GI128 = medfilt2(G128,[5 5]);
end