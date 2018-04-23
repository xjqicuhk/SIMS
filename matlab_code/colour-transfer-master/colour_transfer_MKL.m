%
%   colour transfer algorithm based on linear Monge-Kantorovitch solution 
%
%   IR = colour_transfer_MKL(I_original, I_target, nbiterations);
%
%  (c) F. Pitie 2007
%
%  see reference:
%
%
function IR = colour_transfer_MKL(I0, I1)

if (ndims(I0)~=3)
    error('pictures must have 3 dimensions');
end

X0 = reshape(I0, [], size(I0,3));
X1 = reshape(I1, [], size(I1,3));

A = cov(X0);
B = cov(X1);

T = MKL(A, B);

mX0 = repmat(mean(X0), [size(X0,1) 1]);
mX1 = repmat(mean(X1), [size(X0,1) 1]);

XR = (X0-mX0)*T + mX1;

IR = reshape(XR, size(I0));

function [T] = MKL(A, B)
N = size(A,1);
[Ua,Da2] = eig(A); 
Da2 = diag(Da2); 
Da2(Da2<0) = 0;
Da = diag(sqrt(Da2 + eps));
C = Da*Ua'*B*Ua*Da;
[Uc,Dc2] = eig(C); 
Dc2 = diag(Dc2);
Dc2(Dc2<0) = 0;
Dc = diag(sqrt(Dc2 + eps));
Da_inv = diag(1./(diag(Da)));
T = Ua*Da_inv*Uc*Dc*Uc'*Da_inv*Ua';




