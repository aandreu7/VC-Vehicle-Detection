clearvars,
close all,
clc,


% ====================== CARREGAR IMATGES ======================

folder = './highway/input';
files = dir(fullfile(folder, 'in*.jpg'));

images = cell(1, 1350 - 1051 + 1);
index = 1;

for i = 1:length(files)
    name = files(i).name;
    num = str2double(name(3:8));

    if num >= 1051 && num <= 1350
        filename = fullfile(files(i).folder, name);
        im_color = imread(filename);
        images{index} = rgb2gray(im_color);
        index = index + 1;
    end
end

disp('Carga completa de imágenes.');


% ====================== SEPARACIÓ TRAIN/TEST ======================

im_train = images(1:150);
im_test = images(151:300);


% ====================== CÀLCUL MITJANA I DESVIACIÓ TÍPICA ======================

[M, N] = size(im_train{1});

num_images = size(im_train, 2);

mean_image = zeros(M, N, 'double');
sd_image = zeros(M, N, 'double');    

% Mitjana
for i = 1:num_images
    mean_image = mean_image + double(im_train{i});
end
mean_image = mean_image / num_images;

% Desviació estàndard
for i = 1:num_images
    sd_image = sd_image + (double(im_train{i}) - mean_image).^2;
end
sd_image = sqrt(sd_image / num_images);

mean_image = uint8(mean_image);
sd_image = uint8(sd_image);

figure(1);
imshow(mean_image);
title('Imatge de la mitjana');

figure(2);
imshow(sd_image);
title('Imatge de la desviació estàndard');


