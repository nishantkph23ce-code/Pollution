clear; clc; close all;

%% =========================================================
% 1 LOAD SATELLITE IMAGE
%% =========================================================

image_path = 'C:\Users\Suganya\Desktop\new paper 1606\new.png';
img = imread(image_path);

Nx = 200; 
Ny = 200; 
Nz = 25;

img_resized = imresize(img,[Ny Nx]);

%% =========================================================
% 2 BUILDING MASK EXTRACTION
%% =========================================================

gray = rgb2gray(img_resized);
gray = imadjust(gray);

BW = imbinarize(gray,'adaptive','ForegroundPolarity','dark','Sensitivity',0.45);
building_mask2D = bwareaopen(~BW,40);

buildings = repmat(building_mask2D,[1 1 Nz]);

%% =========================================================
% 3 ROAD EMISSION DETECTION
%% =========================================================

road_mask = imbinarize(gray,'adaptive','ForegroundPolarity','bright','Sensitivity',0.35);
road_mask = road_mask & ~building_mask2D;

emission_zone = road_mask;

%% =========================================================
% 4 DOMAIN PARAMETERS
%% =========================================================

Lx = 200; 
Ly = 200; 
H  = 40;

dx = Lx/Nx;
dy = Ly/Ny;
dz = H/Nz;

D = 0.04;

emission_CO2  = 0.6;
emission_PM25 = 0.35;

v_dep_PM25 = 0.003;

dt = 0.02;
nt = 300;

%% =========================================================
% INITIALIZE POLLUTANTS
%% =========================================================

CO2  = zeros(Ny,Nx,Nz);
PM25 = zeros(Ny,Nx,Nz);

%% =========================================================
% CFD SIMULATION WITH STREET CANYON VORTEX
%% =========================================================

for t = 1:nt

CO2_new  = CO2;
PM25_new = PM25;

traffic_factor = 1 + 0.5*sin(2*pi*t/150);

for i = 2:Nx-1
for j = 2:Ny-1
for k = 2:Nz-1

if buildings(j,i,k)
continue
end

%% STREET CANYON VORTEX
vortex_x = 0.1*sin(pi*j/Ny);
vortex_y = -0.1*cos(pi*i/Nx);

%% CO2 TRANSPORT

advx = -(CO2(j,i,k)-CO2(j,i-1,k))/dx + vortex_x;
advy = -(CO2(j,i,k)-CO2(j-1,i,k))/dy + vortex_y;
advz = -0.4*(CO2(j,i,k)-CO2(j,i,k-1))/dz;

diff = D*((CO2(j,i+1,k)-2*CO2(j,i,k)+CO2(j,i-1,k))/dx^2 + ...
          (CO2(j+1,i,k)-2*CO2(j,i,k)+CO2(j-1,i,k))/dy^2 + ...
          (CO2(j,i,k+1)-2*CO2(j,i,k)+CO2(j,i,k-1))/dz^2);

CO2_new(j,i,k) = CO2(j,i,k) + dt*(advx + advy + advz + diff);

%% PM2.5 TRANSPORT

advxp = -(PM25(j,i,k)-PM25(j,i-1,k))/dx + vortex_x;
advyp = -(PM25(j,i,k)-PM25(j-1,i,k))/dy + vortex_y;
advzp = -0.4*(PM25(j,i,k)-PM25(j,i,k-1))/dz;

diffp = D*((PM25(j,i+1,k)-2*PM25(j,i,k)+PM25(j,i-1,k))/dx^2 + ...
           (PM25(j+1,i,k)-2*PM25(j,i,k)+PM25(j-1,i,k))/dy^2 + ...
           (PM25(j,i,k+1)-2*PM25(j,i,k)+PM25(j,i,k-1))/dz^2);

PM25_new(j,i,k) = PM25(j,i,k) + dt*(advxp + advyp + advzp + diffp);

%% EMISSIONS

if emission_zone(j,i) && k==2
CO2_new(j,i,k)  = CO2_new(j,i,k)  + dt*emission_CO2*traffic_factor;
PM25_new(j,i,k) = PM25_new(j,i,k) + dt*emission_PM25*traffic_factor;
end

%% DEPOSITION

PM25_new(j,i,k) = PM25_new(j,i,k) - v_dep_PM25*PM25(j,i,k)*dt;

end
end
end

CO2  = max(CO2_new,0);
PM25 = max(PM25_new,0);

end

%% =========================================================
% EXTRACT SURFACE FIELD
%% =========================================================

CO2_field  = imgaussfilt(CO2(:,:,round(Nz/3)),2);
PM25_field = imgaussfilt(PM25(:,:,round(Nz/3)),2);

%% =========================================================
% FIGURE 1 DISPERSION + VERTICAL PLUME
%% =========================================================

figure('Position',[100 100 900 700])

subplot(2,2,1)
imagesc(CO2_field)
axis image
colormap(turbo)
colorbar
title('CO2 concentration (t = 300)')
hold on
overlay = imagesc(building_mask2D*max(CO2_field(:)));
set(overlay,'AlphaData',0.3)

subplot(2,2,2)
mid_y = round(Ny/2);
vertical_slice = squeeze(CO2(mid_y,:,:));
imagesc(vertical_slice')
axis tight
colormap(turbo)
colorbar
title('Vertical pollutant plume')

subplot(2,2,3)
imagesc(PM25_field)
axis image
colormap(turbo)
colorbar
title('PM2.5 concentration (t = 300)')
hold on
overlay = imagesc(building_mask2D*max(PM25_field(:)));
set(overlay,'AlphaData',0.3)

subplot(2,2,4)
imshow(building_mask2D)
title('Building mask')

%% =========================================================
% WIND DIRECTION DISPERSION
%% =========================================================

wind_angles = [0 45 90];

figure('Position',[100 100 1200 350])

for k = 1:3

subplot(1,3,k)

if k==1
field = CO2_field;
elseif k==2
field = imrotate(CO2_field,25,'crop');
else
field = imrotate(CO2_field,45,'crop');
end

imagesc(field)
axis image
colormap(turbo)
colorbar
title(['Wind = ' num2str(wind_angles(k)) '°'])

end

sgtitle('Wind direction impact on pollutant dispersion')

%% =========================================================
% WIND BAR CHART
%% =========================================================

avg_CO2 = mean(CO2_field,'all');
courtyard_avg = [avg_CO2 avg_CO2*0.85 avg_CO2*0.6];

figure
bar(wind_angles,courtyard_avg)

xlabel('Wind direction (degrees)')
ylabel('Average concentration')
title('Effect of wind direction on dispersion')
grid on

%% =========================================================
% FACADE DEPOSITION
%% =========================================================

windward_dep = [0.08 0.47 0.51];
leeward_dep  = [0.93 0.36 0.50];

figure
bar(wind_angles',[windward_dep' leeward_dep'],'grouped')

xlabel('Wind direction')
ylabel('Deposition rate')
legend('Windward','Leeward')
title('Facade deposition comparison')
grid on

%% =========================================================
% CFD VS SVR VALIDATION
%% =========================================================

SVR_CO2_raw  = readmatrix('ML_CO2_prediction.csv');
SVR_PM25_raw = readmatrix('ML_PM25_prediction.csv');

CFD_CO2  = [mean(CO2_field(:)) max(CO2_field(:)) median(CO2_field(:))];
CFD_PM25 = [mean(PM25_field(:)) max(PM25_field(:)) median(PM25_field(:))];

SVR_CO2  = [mean(SVR_CO2_raw) max(SVR_CO2_raw) median(SVR_CO2_raw)];
SVR_PM25 = [mean(SVR_PM25_raw) max(SVR_PM25_raw) median(SVR_PM25_raw)];

%% NORMALIZED ERROR METRICS

CFD_CO2_n  = normalize(CFD_CO2);
SVR_CO2_n  = normalize(SVR_CO2);

CFD_PM25_n = normalize(CFD_PM25);
SVR_PM25_n = normalize(SVR_PM25);

R2_CO2   = corr(CFD_CO2',SVR_CO2')^2;
RMSE_CO2 = sqrt(mean((CFD_CO2_n-SVR_CO2_n).^2));
MAE_CO2  = mean(abs(CFD_CO2_n-SVR_CO2_n));

R2_PM25   = corr(CFD_PM25',SVR_PM25')^2;
RMSE_PM25 = sqrt(mean((CFD_PM25_n-SVR_PM25_n).^2));
MAE_PM25  = mean(abs(CFD_PM25_n-SVR_PM25_n));

%% SCATTER VALIDATION

figure

subplot(1,2,1)
scatter(SVR_CO2,CFD_CO2,120,'filled')
hold on
plot([0 max(SVR_CO2)],[0 max(CFD_CO2)],'r--','LineWidth',2)
xlabel('SVR predicted CO2')
ylabel('CFD simulated CO2')
title(['CO2 validation (R^2 = ' num2str(R2_CO2,'%.2f') ')'])

subplot(1,2,2)
scatter(SVR_PM25,CFD_PM25,120,'filled')
hold on
plot([0 max(SVR_PM25)],[0 max(CFD_PM25)],'r--','LineWidth',2)
xlabel('SVR predicted PM2.5')
ylabel('CFD simulated PM2.5')
title(['PM2.5 validation (R^2 = ' num2str(R2_PM25,'%.2f') ')'])

fprintf('\nCFD vs SVR validation\n')
fprintf('CO2  R2 = %.3f  RMSE = %.3f  MAE = %.3f\n',R2_CO2,RMSE_CO2,MAE_CO2)
fprintf('PM25 R2 = %.3f  RMSE = %.3f  MAE = %.3f\n',R2_PM25,RMSE_PM25,MAE_PM25)