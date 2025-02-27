% ====================== CARREGAR IMATGES ======================

folder = './highway';
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

