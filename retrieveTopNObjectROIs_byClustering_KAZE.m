function [topROIs, topScores, topFeatures, topPoints] = ...
    retrieveTopNObjectROIs_byClustering_KAZE(grayImg, ...
        KAZEfeaturesAnchor, KAZEvalidPointsAnchor, params, N)

% Step 1: detect KAZE features in the image
pts = detectKAZEFeatures(grayImg);
if pts.Count < 5
    warning('Not enough KAZE features found.');
    topROIs = []; topScores = []; topFeatures = {}; topPoints = {}; return;
end
[feat, validPts] = extractFeatures(grayImg, pts);

% Step 2: cluster keypoints spatially (K-means)
numClusters = params.numClusters;
coords = validPts.Location; % Nx2 matrix (x,y)
[idx, C] = kmeans(coords, numClusters, 'MaxIter', 200, 'Replicates', 3);

% Step 3: compute similarity score per cluster
bboxes = [];
scores = [];
features = {};
points = {};

for c = 1:numClusters
    clusterPts = coords(idx == c, :);
    if size(clusterPts,1) < 5
        continue; % skip small clusters
    end

    % compute bounding box of this cluster
    minX = min(clusterPts(:,1)); maxX = max(clusterPts(:,1));
    minY = min(clusterPts(:,2)); maxY = max(clusterPts(:,2));
    bbox = [minX, minY, maxX-minX, maxY-minY];

    % extract ROI and its features
    roi = imcrop(grayImg, bbox);

    % detect + extract again for local context (optional)
    ptsROI = detectKAZEFeatures(roi);
    if ptsROI.Count < 3, continue; end
    [featROI, validPtsROI] = extractFeatures(roi, ptsROI);

    % compute explicit similarity score
    score = computeSimilarityScore_KAZE_explicit(KAZEfeaturesAnchor, featROI, params);

    bboxes = [bboxes; bbox];
    scores = [scores; score];
    features{end+1} = featROI;
    points{end+1} = validPtsROI;
end

% Step 4: rank by similarity and return top N
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
