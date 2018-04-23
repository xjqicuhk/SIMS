%This package contains two scripts to run colour grading as described in
%
%[Pitie07] Automated colour grading using colour distribution transfer. 
%          F. Pitie , A. Kokaram and R. Dahyot (2007) 
%          Computer Vision and Image Understanding. 
%[Pitie05] N-Dimensional Probability Density Function Transfer and its 
%          Application to Colour Transfer. F. Pitie , A. Kokaram and 
%          R. Dahyot (2005) In International Conference on Computer Vision 
%          (ICCV'05). Beijing, October.
%
% The grain reducer technique is not provided here.
%
% Note that both pictures are copyrighted.
%
% send an email to fpitie@mee.tcd.ie if you want more information

fprintf('Colour transfer demo based on:\n');
fprintf('  [Pitie05a] Pitie et al. N-Dimensional Probability Density Function Transfer and its Application to Colour Transfer. ICCV05.\n');
fprintf('  [Pitie05b] Pitie et al. Towards Automated Colour Grading. CVMP05.\n');
fprintf('  [Pitie07a] Pitie et al. Automated colour grading using colour distribution transfer. CVIU. 2007.\n');
fprintf('  [Pitie07b] Pitie et al. The linear Monge-Kantorovitch linear colour mapping for example-based colour transfer. CVMP07.\n');
fprintf('  [Pitie08]  Pitie et al. Enhancement of Digital Photographs Using Color Transfer Techniques. Single-Sensor Imaging. 2008.\n');

fprintf('  ... load images\n');

I0 = double(imread('scotland_house.png'))/255;
I1 = double(imread('scotland_plain.png'))/255;

fprintf('  ... MKL Colour Transfer \n');

IR_mkl = colour_transfer_MKL(I0,I1);

fprintf('  ... seed the random number generator\n');
rng(0);

fprintf('  ... IDT Colour Transfer (slow implementation) \n');
IR_idt = colour_transfer_IDT(I0,I1,10);

fprintf('  ... regrain post-processing on IDT results \n');
IR_idt_regrain = regrain(I0,IR_idt);

fprintf('  [ok] \n');

% display results
screensize = get(0,'ScreenSize');
sz = [576, 1024];
figure('Position', [ ceil((screensize(3)-sz(2))/2), ceil((screensize(4)-sz(1))/2), sz(2), sz(1)]);
subplot('Position',[0.01  0.4850 0.3200 .47]); 
imshow(I0); title('Original Image'); 

subplot('Position',[0.3400  0.4850 0.3200 .47]); 
imshow(I1); title('Target Palette'); 

subplot('Position',[0.01 0.01 0.3200 .47]); 
imshow(IR_mkl); title('Result After MKL Colour Transfer [Pitie07b]'); 

subplot('Position',[0.3400 0.01 0.3200 .47]);  
imshow(IR_idt); title('Result After IDT Colour Transfer [Pitie05a,Pitie05b,Pitie07a]'); 

subplot('Position',[0.6700 0.01 0.3200 .47]); 
imshow(IR_idt_regrain); title('After IDT and Regrain [Pitie05b,Pitie07a]'); 


