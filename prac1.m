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

%% ====================== SEGMENTACIÓ BÀSICA ======================

thr = 40;

segmentation_images = cell(1, size(images, 2));
for x=1:size(images, 2)
    segmentation_images{x} = abs(images{x} - mean_image) > thr;
end

% Imatge de prova
figure(3);
imshow(segmentation_images{7});
title('Segmentació bàsica')


%% ====================== SEGMENTACIÓ AVANÇADA ======================

% Primera aproximació (1 si |Ii,j − µi,j | > ασi,j + β, 0 en cas contrari)

a = 0.15 * sd_image;
b = 5;

filter_image = sd_image;
filter_image(filter_image < 35) = 130; % Blanquejem la imatge per millorar els resultats (opcional)

threshold = a + b;
adjusted_mean = mean_image - filter_image;

segmentation_images = cell(1, size(images, 2));
for x = 1:size(images, 2)
    diff_image = abs(images{x} - adjusted_mean) > threshold;
    segmentation_images{x} = adjusted_mean - uint8(diff_image * 255); %im2uint8(diff_image); %
    segmentation_images{x}(segmentation_images{x} > b) = 255;
end


% Imatge de prova
figure(4);
subplot(1, 2, 1);
imshow(segmentation_images{7});
subplot(1, 2, 2);
imshow(images{7});
sgtitle('Segmentació avançada: primera aproximació');

%% ================================ OPENING ===========================

SE = strel('rectangle', [3, 3]);

SE2 = strel('rectangle', [3, 3]); 


opened_images = cell(1, size(images, 2));
for x = 1:size(images, 2)
    opened_images{x} = imdilate(imerode(segmentation_images{x},SE),SE2);
end
figure(5)
subplot(1, 2, 1);
imshow(segmentation_images{7});
subplot(1, 2, 2);
imshow(opened_images{7});

%% ============================== GRAVAR VIDEO ==========================
video = VideoWriter('video_output.avi', 'Motion JPEG AVI');
video.FrameRate = 20;  
open(video);

for i = 1:length(opened_images)
    writeVideo(video, opened_images{i});
end
close(video);