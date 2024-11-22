function [tX,fY]=pyMatlibMusicNoDown(realTheta,realFai,path,resPath)
    %% 若用于测试的时候，会修改data_path和error_path，在跑实验数据的时候要记得修改
    pplot = false;

    soundF = 40e3;%Hz
    % 这一过程主要是用于模拟实验过程，soundF表示产生的信号频率
    DsoundF = abs(soundF-floor(soundF/3000+0.5)*3000);% sound frequency after downsample to 3kHz
    l=340/soundF; %wave length	
    r = 0.002; % rad
    count=1;
    data_path = [path,'testdata.txt'];
%     data_path = [path,'validatedata.txt'];
%     data_path = "testdata.txt";
    
    
    data=load(data_path);
    signal1=data;
    signal1 = signal1';
    %% get complex sequence of the nerrow band signal.
    signal3000 = signal1;
    
%     figure(1);
%     t=linspace(1,900,900);
%     plot(t,signal1(1,1:900))
%     fs = 1200e3; % 初始的采样频率，后面会降采样为3khz，这样是为了模拟采样的过程
%     t=0:1/fs:0.1; % 产生两秒的时间，时间间隔为1/fs
%     reduceSig = sin(2*pi*soundF*(ones(4,1)*t));% Hz
%     reduceSig = reduceSig(:,1:900);
%     signal3000 = signal3000+5*reduceSig;

    
%     noise = randn(4,900);
%     signal3000 = signal3000+noise;
    
    [signalC,N_x] = getCmx(signal3000(:,100:800),60,10,1200e3,40e3);%window length better is times of 3000/DsoundF for simulation
%     [signalC,N_x] = getCmx(signal3000(:,100:800),60,10,3000,DsoundF);%window length better is times of 3000/DsoundF for simulation

    R=1/N_x*(signalC*signalC.');
    [V,D]=eig(R);
    [lambda,index] = sort((diag(D)));
    UU=V(:,index(1:3));

    thetaX = 0:90;
    faiX = 0:180;
    [ Pmusic,tX,fY ] = searchP( thetaX,faiX,UU,l,r );
    thetaX = tX-5:0.1:tX+5;
    faiX = fY-5:0.1:fY+5;
    [ Pm,tX,fY ] = searchP( thetaX,faiX,UU,l,r );
    thetaX = tX-0.5:0.01:tX+0.5;
    faiX = fY-0.5:0.01:fY+0.5;
    [ Pm,tX,fY ] = searchP( thetaX,faiX,UU,l,r );


    if pplot
        theta_x=0:90;
        fai_x=0:180;
        figure(3)
        mesh(fai_x,theta_x,Pmusic);
        title('MUSIC spatial spectrum');
        xlabel('azimuth'); 
        ylabel('elevation');
        zlabel('angle');
        grid on; 
        figure(4)
        plot(fai_x,Pmusic)
        xlabel('azimuth fai');
        ylabel('signal power');
        figure(5)
        plot(theta_x,Pmusic);
        xlabel('elevation theta');
    end
    
    mse = zeros(2,100);
    mse(:,count) = [abs(realTheta-tX);abs(realFai-fY)];
    
    AA = mse(:,count);
    err1 = AA(1);
    err2 = AA(2);
    RMSE = sqrt((err1^2+err2^2)/2);
    RMSE_str = num2str(RMSE);
    
    perr(4,count) = mean(mse(:,count));
    error = num2str(perr(4,count));
    disp(['theta=', num2str(tX),' | fai=',num2str(fY), ' | error=',error]);   
    disp(['RMSEerror=',RMSE_str]); 
    % pause(1000)
    error_path = [resPath,'error.txt'];
    %%error_path = 'error.txt';
    fid = fopen(error_path,'a');
    fprintf(fid,num2str(error));
    fprintf(fid,'\n');
    fclose(fid);
    
end
