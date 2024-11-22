function [ out , N_x] = getCmx( in,winLen,winStep, SamF,SouF )
t = 0:1/SamF:(winLen-1)/SamF;
c = cos(2*pi*SouF*t)';
s = sin(2*pi*SouF*t)';
[x,y] = size(in);
out = zeros(x,(floor((y-winLen)/winStep)+1));
N_x = (floor((y-winLen)/winStep)+1);

for i = 1:N_x    
    win = in(:,(i-1)*winStep+1:(i-1)*winStep+winLen);
    Rwin = win*c;
    Iwin = win*s;
    out(:,i) = Rwin+1j*Iwin;
end

end

