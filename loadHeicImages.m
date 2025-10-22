function images = loadHeicImages(directory)
%LOADHEICIMAGES loads all the images with .png extension in the given directory
%   Inputs:     directory - string containing the full path where images are
%                           contained, separated by "/"
%   Outputs:    images - cell array containing only .heic images in the
%                        given directory
files = dir(fullfile(directory,"*.heic"));
images = cell(length(files),1);

for i = 1:length(files)
    filePath = fullfile(directory,files(i).name);
    images{i} = imread(filePath);
end
