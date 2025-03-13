clearvars,
close all,
clc,


% ====================== CARREGAR IMATGES ======================

folder = './highway/input';
files = dir(fullfile(folder, 'in*.jpg'));
folder_gt = './highway/groundtruth';
files_gt = dir(fullfile(folder_gt, 'gt*.png'));

images = cell(1, 1350 - 1051 + 1);
images_gt = cell(1, 1350 - 1051 + 1);

index = 1;

for i = 1:length(files)
    name = files(i).name;
    num = str2double(name(3:8));

    if num >= 1051 && num <= 1350
        filename = fullfile(files(i).folder, name);
        im_color = imread(filename);
        images{index} = rgb2gray(im_color);

        filename_gt = fullfile(folder_gt, sprintf('gt%06d.png', num));
        images_gt{index} = imread(filename_gt);

        index = index + 1;
    end
end

disp('Carga completa de imágenes.');


% ====================== SEPARACIÓ TRAIN/TEST ======================

im_train = images(1:150);
im_test = images(151:300);


% ====================== CÀLCUL MITJANA I DESVIACIÓ TÍPICA ======================

% Mitjana
images_stack = cat(3, im_train{:});
mean_image = mean(double(images_stack), 3);

% Desviació estàndard
sd_image = std(double(images_stack), 0, 3);

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

segmentation_images = cell(1, size(im_train, 2));
for x=1:size(im_train, 2)
    segmentation_images{x} = abs(im_train{x} - mean_image) > thr;
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

segmentation_images = cell(1, size(im_train, 2));
for x = 1:size(im_train, 2)
    diff_image = abs(im_train{x} - adjusted_mean) > threshold;
    segmentation_images{x} = adjusted_mean - uint8(diff_image * 255);
    segmentation_images{x}(segmentation_images{x} > b) = 255;
end


% Imatge de prova
figure(4);
subplot(1, 2, 1);
imshow(segmentation_images{7});
subplot(1, 2, 2);
imshow(im_train{7});
sgtitle('Segmentació avançada: primera aproximació');

%% ================================ OPENING ===========================
 
 SE = strel("disk",1);
 
 SE2 = strel('diamond',1);
 
 opened_images = cell(1, size(im_train, 2));
 for x = 1:size(im_train, 2)
     opened_images{x} = imdilate(imerode(segmentation_images{x},SE),SE2);
 end
 figure(5)
 subplot(1, 2, 1);
 imshow(segmentation_images{7});
 subplot(1, 2, 2);
 imshow(opened_images{7});
 sgtitle('Opening: Before and after');


%% ========================= AVALUACIÓ ====================================

accuracy = zeros(1, length(opened_images));
precision = zeros(1, length(opened_images));
recall = zeros(1, length(opened_images));
f1_score = zeros(1, length(opened_images));
sensitivity = zeros(1, length(opened_images));
specificity = zeros(1, length(opened_images));

for i = 1:length(opened_images)
    segmented = opened_images{i};
    gt = images_gt{i};

    gt = uint8(gt > 0) * 255;

    TP = sum(segmented(:) == 1 & gt(:) == 255);
    TN = sum(segmented(:) == 0 & gt(:) == 0);
    FP = sum(segmented(:) == 1 & gt(:) == 0);
    FN = sum(segmented(:) == 0 & gt(:) == 255);

    total_pixels = numel(segmented);
    correct_predictions = TP + TN;
    
    accuracy(i) = correct_predictions / total_pixels;
end

mean_accuracy = mean(accuracy);

fprintf('Accuracy mitjà: %.4f\n', mean_accuracy);


