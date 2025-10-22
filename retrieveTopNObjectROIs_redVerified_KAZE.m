function [topROIs, topScores, topFeatures, topPoints] = ...
    retrieveTopNObjectROIs_redVerified_KAZE(grayImg, colorImg, ...
        KAZEfeaturesAnchor, KAZEvalidPointsAnchor, params, N, redThresh, minRedFraction)

% Step 1: detect candidate ROIs via edges
edges = edge(grayImg, 'Canny',[0.05 0.2]);

cc = bwconncomp(edges);
stats = regionprops(cc, 'BoundingBox', 'Area');

bboxes = [];
scores = [];
features = {};
points = {};

% Precompute HSV for red verification
hsvImg = rgb2hsv(colorImg);
hue = hsvImg(:,:,1);
sat = hsvImg(:,:,2);
val = hsvImg(:,:,3);
maskRed = ((hue < redThresh) | (hue > 1-redThresh)) & (sat > 0.25) & (val > 0.2);

for i = 1:length(stats)
    bbox = stats(i).BoundingBox;
    if stats(i).Area < 100, continue; end
    roi = imcrop(grayImg, bbox);
    roiMask = imcrop(maskRed, bbox);
    redFraction = sum(roiMask(:)) / numel(roiMask);

    % verify there is enough red hue
    if redFraction < minRedFraction
        continue;
    end

    % extract KAZE features
    ptsROI = detectKAZEFeatures(roi);
    if ptsROI.Count < 3, continue; end
    [featROI, validPtsROI] = extractFeatures(roi, ptsROI);

    % compute similarity explicitly
    score = computeSimilarityScore_KAZE_explicit(KAZEfeaturesAnchor, featROI);

    bboxes = [bboxes; bbox];
    scores = [scores; score];
    features{end+1} = featROI;
    points{end+1} = validPtsROI;
end

if isempty(scores)
    topROIs = []; topScores = []; topFeatures = {}; topPoints = {}; return;
end

[sortedScores, idx] = sort(scores, 'descend');
n = min(N, numel(sortedScores));
topROIs = bboxes(idx(1:n), :);
topScores = sortedScores(1:n);
topFeatures = features(idx(1:n));
topPoints = points(idx(1:n));
end
