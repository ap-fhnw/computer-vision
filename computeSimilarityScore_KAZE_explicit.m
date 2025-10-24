function scoreMean = computeSimilarityScore_KAZE_explicit(KAZEfeaturesAnchor, featROI, params)
% computeSimilarityScore_KAZE_explicit
% Computes mean similarity score between ROI and 3 anchor KAZE features.
%
% Inputs:
%   KAZEfeaturesAnchor : {f1, f2, f3}, each an MxN descriptor matrix
%   featROI            : PxN descriptor matrix (ROI)
%   params             : dynamically set similarity function
%
% Output:
%   scoreMean : mean similarity across 3 anchors


if isempty(KAZEfeaturesAnchor) || isempty(featROI)
    scoreMean = 0;
    return;
end

% --- Normalize candidate descriptor type ---
if isa(featROI, 'binaryFeatures')
    featROI = double(featROI.Features);
elseif iscell(featROI)
    featROI = double(featROI{1}.Features);
end

% --- Iterate through anchors ---
numAnchors = numel(KAZEfeaturesAnchor);
anchorScores = zeros(1, numAnchors);

for i = 1:numAnchors
    anchor = KAZEfeaturesAnchor{i};

    if isa(anchor, 'binaryFeatures')
        descAnchor = double(anchor.Features);
    elseif isnumeric(anchor)
        descAnchor = double(anchor);
    else
        error('Unsupported anchor descriptor type: %s', class(anchor));
    end

    numA = size(descAnchor,1);
    numC = size(featROI,1);
    simVals = zeros(numA,numC);

    for a = 1:numA
        descA = reshape(descAnchor(a,:),[],1);
        for c = 1:numC
            descC = reshape(featROI(c,:),[],1);
            simVals(a,c) = params.similarityFunc(descA, descC);
        end
    end

    if params.expectHighScore
        anchorScores(i) = max(simVals(:));
    else
        anchorScores(i) = 1 / (1 + min(simVals(:)));
    end
end

% Final similarity = mean across all 3 anchors
scoreMean = mean(anchorScores);
end