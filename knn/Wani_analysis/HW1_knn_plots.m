k = [1 3 5 9 15]; % row
limit = [50 100 200 300 500]; % column

accuracy = [0.643300 0.692300 0.764400 0.810200 0.845800;
    0.589500 0.670000 0.738200 0.789000 0.831100;
    0.490900 0.608100 0.690200 0.747700 0.802800;
    0.402300 0.571100 0.656800 0.722600 0.789700;
    0.311000 0.513700 0.618000 0.683400 0.760100];


plot(k,accuracy, '.-', 'linewidth', 1.2, 'MarkerSize',20);

set(gcf, 'color', 'w');
set(gca, 'fontsize', 18, 'linewidth', 1.2, 'xlim', [0, 16], 'ylim', [.25 .9]);
xlabel('k');
ylabel('Accuracy');

figdir = '/Users/clinpsywoo/github/ml-hw/knn/Wani_analysis';
cd(figdir);

try
    pagesetup(gcf);
    saveas(gcf, 'figure.pdf');
catch
    pagesetup(gcf);
    saveas(gcf, 'figure.pdf');
end


%%

sepplot(k,accuracy)


    
conf = [7,9;
    9,4;
    5,3;
    4,9;
    4,9;
    7,9;
    5,3;
    5,3;
    5,3;
    4,9;
    7,9;
    2,1;
    7,9;
    5,3;
    4,9;
    7,9;
    2,1;
    7,9;
    5,3;
    2,1;
    7,9;
    2,1;
    7,9;
    2,1;
    2,1];
% 7 & 9 - 8 times. 
    
    