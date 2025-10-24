function funcs = similarity()
% similarity - Registry and container for all similarity measures.
%
% Usage:
%   funcs = similarity();
%   score = funcs.zncc(patch1, patch2);

funcs = struct( ...
    'sad',  @sad, ...
    'zsad', @zsad, ...
    'ssd',  @ssd, ...
    'zssd', @zssd, ...
    'ncc',  @ncc, ...
    'zncc', @zncc ...
);

end

% --- Similarity functions below ---

function s = sad(I1, I2)
% Sum of Absolute Differences
s = sum(abs(I1(:) - I2(:)));
end

function s = zsad(I1, I2)
% Zero-mean SAD (brightness invariant)
I1 = I1 - mean(I1(:));
I2 = I2 - mean(I2(:));
s = sum(abs(I1(:) - I2(:)));
end

function s = ssd(I1, I2)
% Sum of Squared Differences
d = I1 - I2;
s = sum(d(:).^2);
end

function s = zssd(I1, I2)
% Zero-mean SSD (brightness invariant)
I1 = I1 - mean(I1(:));
I2 = I2 - mean(I2(:));
d = I1 - I2;
s = sum(d(:).^2);
end

function s = ncc(I1, I2)
% Normalized Cross-Correlation
num = sum(I1(:) .* I2(:));
den = sqrt(sum(I1(:).^2) * sum(I2(:).^2));
s = num / den;
end

function s = zncc(I1, I2)
% Zero-mean Normalized Cross-Correlation (brightness + contrast invariant)
I1 = I1 - mean(I1(:));
I2 = I2 - mean(I2(:));
num = sum(I1(:) .* I2(:));
den = sqrt(sum(I1(:).^2) * sum(I2(:).^2));
s = num / den;
end
