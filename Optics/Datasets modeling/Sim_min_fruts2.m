%% Ghost imaging reconstraction
clc; close all; clear;
img_size = 64; RN = 1000; % Число измерений (итераций GI)
folder_info = dir('E:\vova shumigay\1. Работа\Задачи\2025\SWIR camera\Photos\min256\*.jpg'); 

for i = 1:length(folder_info)
    img = imread(fullfile(folder_info(i).folder,folder_info(i).name)); img = im2double(img); img_gray = rgb2gray(img);
    IMG4CONTR = imresize(img_gray, [img_size img_size]); 
    img_contr2 = imadjust(IMG4CONTR, [0.2 0.9], [0 1]); %img_noise2 = imnoise(bluring_img(img_contr2, 30, randi([7, 12])),'gaussian',0.2,0.01);
%     img_contr3 = imadjust(IMG4CONTR, [0.1 0.8], [0 1]); img_noise3 = imnoise(bluring_img(img_contr3, 70, randi([9, 15])),'gaussian',-0.3,0.1);
    img_contr1 = img_contr2; img_contr3 = img_contr2;
    
    img_GI1 = white_noise_shift_rec(img_size, 2, img_contr1, 5500); img_GI2 = white_noise_shift_rec(img_size, 2, img_contr2, 7000);
    img_GI3 = white_noise_shift_rec(img_size, 2, img_contr3, 5700);%pink_noise_rec(img_size, img_noise3, 800);
    
    img_noise1 = imadjust(img_GI1, [0 1], [0 1]); img_noise2 = imadjust(img_GI2, [0.1 0.9], [0 1]); img_noise3 = imadjust(img_GI3, [0.1 0.9], [0 1]);    

%     figure; tiledlayout(3,4); nexttile; imshow(IMG4CONTR); nexttile; imshow(img_contr1, []); nexttile; imshow(img_contr2,[]); nexttile; imshow(img_contr3, []);
%     nexttile; imshow(img); nexttile; imshow(img_noise1, []); nexttile; imshow(img_noise2,[]); nexttile; imshow(img_noise3, []);
%     nexttile; imshow(img); nexttile; imshow(img_GI1, []); nexttile; imshow(img_GI2,[]); nexttile; imshow(img_GI3, []);

    len = Length_of_Number(i); number_of_zeros = Length_of_Number(length(folder_info)) - len; zero_top = string();
    for k = 1:number_of_zeros
        zero_top = zero_top + string(0);
    end
    imwrite(img_noise1, ['Dataset\Minerals\64\800\',num2str(zero_top), num2str(i),'_64_800.jpg'])
    imwrite(img_noise2, ['Dataset\Minerals\64\1050\',num2str(zero_top), num2str(i),'_64_1050.jpg'])
    imwrite(img_noise3, ['Dataset\Minerals\64\1550\',num2str(zero_top), num2str(i),'_64_1550.jpg'])   
end