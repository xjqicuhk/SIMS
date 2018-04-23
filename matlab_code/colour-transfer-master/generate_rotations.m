% rotations = find_all(ndim, NbRotations)
%
% ndim = 2 or 3 (but the code can be changed)
%
%
% code for generating an optimised sequence of rotations
% for the IDT pdf transfer algorithm
% although the code is not beautiful, it does the job.
%
function rotations = generate_rotations(ndim, NbRotations)

if (ndim == 2)
    l = [0 pi/2];
elseif (ndim == 3)
    l = [0 0 pi/2 0 pi/2 pi/2];
else % put here initialisation for higher orders
end

fprintf('rotation ');
for i = 1:(NbRotations-1)
    fprintf('%d ...', i );
    l = [l ; find_next(l, ndim)]; %l(end,:)+ones(1,ndim-1)*pi/2)]
    fprintf('\b\b\b', i );
end

M = ndim;

rotations = cell(1,NbRotations);
for i=1:size(l, 1)
    for j=1:M
        b_prev(j,:) = hyperspherical2cartesianT(l(i,(1:ndim-1) + (j-1)*(ndim-1)));
    end
    b_prev = grams(b_prev')';
    rotations{i} = b_prev;
end


end
%
%
function [x] = find_next( list_prev_x, ndim)

prevx = list_prev_x; % in hyperspherical coordinates
nprevx = size(prevx,1);
hdim = ndim - 1;
M = ndim;

% convert points to cartesian coordinates
c_prevx = zeros(nprevx*M, ndim);
c_prevx = [];
for i=1:nprevx
    for j=1:M
        b_prev(j,:) = hyperspherical2cartesianT(prevx(i,(1:hdim) + (j-1)*hdim));
    end
    b_prev = grams(b_prev')';
    c_prevx = [c_prevx; b_prev];
end

c_prevx;

options = optimset('TolX', 1e-10);
options = optimset(options,'Display','off');

minf = inf;
for i=1:10
    x0 = rand(1, hdim*M)*pi - pi/2;
    x = fminsearch(@myfun, x0, options);
    f = myfun(x);
    if f < minf
        minf = f;
        mix = x;
    end
end

%%
% f - Compute the function value at x
    function [f] = myfun(x)
        % compute the objective function
        c_x = zeros(M, ndim);
        for i=1:M
            c_x(i, :) = hyperspherical2cartesianT(x((1:hdim) + (i-1)*hdim));
        end
        c_x = grams(c_x')';
        f = 0;
        for i=1:M
            for p=1:size(c_prevx, 1)
                d = (c_prevx(p,:) - c_x(i, :)) * (c_prevx(p,:) - c_x(i, :))';
                f = f + 1/(1 + d);
                d = (c_prevx(p,:) + c_x(i, :)) * (c_prevx(p,:) + c_x(i, :))';
                f = f + 1/(1 + d);
            end
        end
    end
%%

end


%
%
function c = hyperspherical2cartesianT(x)

c = zeros(1, length(x)+1);
sk = 1;
for k=1:length(x)
    c(k) = sk*cos(x(k));
    sk = sk*sin(x(k));
end
c(end) = sk;

end

% Gram-Schmidt orthogonalization of the columns of A.
% The columns of A are assumed to be linearly independent.
function [Q, R] = grams(A)

[m, n] = size(A);
Asave = A;
for j = 1:n
    for k = 1:j-1
        mult = (A(:, j)'*A(:, k)) / (A(:, k)'*A(:, k));
        A(:, j) = A(:, j) - mult*A(:, k);
    end
end
for j = 1:n
    if norm(A(:, j)) < sqrt(eps)
        error('Columns of A are linearly dependent.')
    end
    Q(:, j) = A(:, j) / norm(A(:, j));
end
R = Q'*Asave;
end

