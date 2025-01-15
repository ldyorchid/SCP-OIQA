% Demo script
% Uncomment each case to see the results
SRC = 'F:\OIQA\saliency';
file = dir(fullfile(SRC));
srcSuffix = '.jpg';
outputSize = [224, 224];
for num=3:length(file)
    filename=file(num).name;
    angle=dir(fullfile(SRC,filename));
    for x=3:length(angle)
        viewport=angle(x).name;
        files = dir(fullfile(SRC,filename,viewport, strcat('*', srcSuffix)));
        for k=1:length(files)
            disp(k);
            srcName = files(k).name;            
            I = imread(fullfile(SRC,filename,viewport, srcName));
            z = floor((size(I, 2) - outputSize(2)) / 2) + 1;
            y = floor((size(I, 1) - outputSize(1)) / 2) + 1;
            I = imcrop(I, [z, y, outputSize(2)-1, outputSize(1)-1]);
            S = tsmooth(I,0.005,1);
            imwrite(S, fullfile(SRC,filename,viewport, srcName));
        end
    end
end

S = tsmooth(I,0.005,1);
figure, imshow(I), figure, imshow(S);

% I = (imread('imgs/graffiti.jpg'));
% S = tsmooth(I,0.015,3);
% figure, imshow(I), figure, imshow(S);

% I = (imread('imgs/crossstitch.jpg'));
% S = tsmooth(I,0.015,3);
% figure, imshow(I), figure, imshow(S);


% I = (imread('imgs/mosaicfloor.jpg'));
% S = tsmooth(I, 0.01, 3, 0.02, 5);
% figure, imshow(I), figure, imshow(S);






