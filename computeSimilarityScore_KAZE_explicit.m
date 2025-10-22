function scoreMean = computeSimilarityScore_KAZE_explicit(KAZEfeaturesAnchor, featROI)
% computeSimilarityScore_KAZE_explicit
% Computes mean similarity score between ROI and 3 anchor KAZE features.
%
% Inputs:
%   KAZEfeaturesAnchor : {f1, f2, f3}, each an MxN descriptor matrix
%   featROI            : PxN descriptor matrix (ROI)
%
% Output:
%   scoreMean : mean similarity across 3 anchors

numAnchors = numel(KAZEfeaturesAnchor);
anchorScores = zeros(1, numAnchors);

for i = 1:numAnchors
    fA = double(KAZEfeaturesAnchor{i});
    fR = double(featROI);

    % Normalize each feature vector to unit length
    fA = fA ./ vecnorm(fA,2,2);
    fR = fR ./ vecnorm(fR,2,2);

    % Compute cosine similarity matrix
    simMatrix = fA * fR'; % [M_anchor x P_roi]

    % take the best match per anchor feature
    maxPerAnchor = max(simMatrix, [], 2);

    % average similarity for this anchor
    anchorScores(i) = mean(maxPerAnchor);
end

% Final similarity = mean across all 3 anchors
scoreMean = mean(anchorScores);
end