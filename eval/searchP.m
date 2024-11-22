function [ Pm,maxTheta,maxFai ] = searchP( thetaX,faiX,UU,l,r )
degrade=pi/180; %degree to radio
Pm = zeros(length(thetaX),length(faiX));
NN = UU';

U4U = UU*NN;
for theta = 1:length(thetaX)
for fai = 1:length(faiX)
AA=[exp(-1j*2*pi*r*cos(thetaX(theta)*degrade)*cos(faiX(fai)*degrade)/l);... %(r,0)
    exp(-1j*2*pi*(-r)*cos(thetaX(theta)*degrade)*cos(faiX(fai)*degrade)/l);... %(-r,0)
    exp(-1j*2*pi*r*sin(thetaX(theta)*degrade)*cos(faiX(fai)*degrade)/l); ... %(0,r)
    exp(-1j*2*pi*(-r)*sin(thetaX(theta)*degrade)*cos(faiX(fai)*degrade)/l)]; %(0,-r)

%%AT = AA';
P=AA'*AA;
AAA=AA'*U4U*AA;
Pm(theta,fai)=abs(P./AAA);
end
end
Cmax=max(Pm);
Mmax=max(Cmax);
[X,Y]=find(Pm==Mmax);
maxTheta = thetaX(X);
maxFai = faiX(Y);
end

