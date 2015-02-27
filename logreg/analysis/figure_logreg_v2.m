%% Number 1: step size
basedir = '/Users/clinpsywoo/github/ml-hw/logreg/analysis';
load(fullfile(basedir, 'data.mat'));

datname = {'dat10', 'dat05', 'dat01', 'dat001', 'dat0001', 'dat_var'};
cols = [0.6196    0.0039    0.2588
    0.8353    0.2431    0.3098
    0.9569    0.4275    0.2627
    0.9922    0.6824    0.3804
    0.9961    0.8784    0.5451
    0.1961    0.5333    0.7412];

create_figure('analysis1', 1,2);

subplot(1,2,1);
for i = 1:numel(datname)
    eval(['plot(' datname{i} '(:,1), ''color'', cols(i,:), ''linewidth'', 2);']);
    hold on;
    
end
set(gca, 'linewidth', 1.2, 'ylim', [.5 1], 'xlim', [0 215], 'fontsize', 20);

subplot(1,2,2);
for i = 1:numel(datname)
    eval(['plot(' datname{i} '(:,2), ''color'', cols(i,:), ''linewidth'', 2);']);
    hold on;
end

set(gca, 'linewidth', 1.2, 'ylim', [.5 1], 'xlim', [0 215], 'fontsize', 20);

cd(basedir);

try
    pagesetup(gcf);
    saveas(gcf, 'figure1.pdf');
catch
    pagesetup(gcf);
    saveas(gcf, 'figure1.pdf');
end


%% passes
%1: TA 0.965226	HA 0.932331
%2: TA 0.989662	HA 0.939850
%3: TA 0.996241	HA 0.947368
%4: TA 0.997180	HA 0.954887
%5: TA 0.999060	HA 0.939850

dat_passes = [0.965226	0.932331
0.989662	0.939850
0.996241	0.947368
0.997180	0.954887
0.999060	0.939850];

create_figure('analysis2');
plot(1:5,dat_passes(:,1), '.-', 'linewidth', 2, 'MarkerSize',25, 'color', [0.6196    0.0039    0.2588]);
plot(1:5,dat_passes(:,2), '.-', 'linewidth', 2, 'MarkerSize',25, 'color', [0.1961    0.5333    0.7412]);

set(gcf, 'position', [-1327         354         515         352]);
set(gca, 'fontsize', 20, 'linewidth', 1.2, 'xlim', [0.8, 5.2], 'ylim', [.9 1]);

xlabel('Passes');
ylabel('Accuracy');

basedir = '/Users/clinpsywoo/github/ml-hw/logreg/analysis';
cd(basedir); 

try
    pagesetup(gcf);
    saveas(gcf, 'figure2.pdf');
catch
    pagesetup(gcf);
    saveas(gcf, 'figure2.pdf');
end


%% 3 and 4

% best word for base:hit
% worst word for base:tandem
% best word for hockey:hockey
% worst word for base:kicked

%     print 'best word for base:' + vocab[lr.beta.argmax()]
%     base = where(lr.beta>0)
%     base = base[0]
%     print 'worst word for base:' + vocab[base[lr.beta[base].argmin()]]
%     print 'best word for hockey:' + vocab[lr.beta.argmin()]
%     hockey = where(lr.beta<0)
%     hockey = hockey[0]
%     print 'worst word for base:' + vocab[hockey[lr.beta[hockey].argmax()]]

%% extra

