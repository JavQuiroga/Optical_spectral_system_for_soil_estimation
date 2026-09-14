%% GRAFICAR_SELECTED_X_VS_LONGITUD_ONDA
% Primera etapa del analisis de separabilidad espectral.
% Grafica los valores medidos de selected_x en funcion de la longitud de
% onda. En este paso no se corrigen datos ni se seleccionan bandas.

clear;
close all;
clc;

%% Rutas
scriptPath = mfilename('fullpath');
scriptDir = fileparts(scriptPath);

inputCsv = fullfile(scriptDir, '..', ...
    'spectral_characterization_plots', 'spectral_characterization.csv');
outputDir = fullfile(scriptDir, 'resultados');
lambdaMaxNm = 1471;

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% Leer los datos
if ~isfile(inputCsv)
    error('No se encontro el CSV de entrada: %s', inputCsv);
end

T = readtable(inputCsv, 'VariableNamingRule', 'preserve');

requiredColumns = {'wavelength_nm', 'selected_x'};
for k = 1:numel(requiredColumns)
    if ~ismember(requiredColumns{k}, T.Properties.VariableNames)
        error('Falta la columna requerida "%s" en el CSV.', ...
            requiredColumns{k});
    end
end

wavelengthNm = double(T.wavelength_nm);
selectedX = double(T.selected_x);

% Se eliminan solamente filas sin valores numericos validos y se aplica el
% limite superior acordado para el intervalo util.
valid = isfinite(wavelengthNm) & isfinite(selectedX) & ...
    wavelengthNm <= lambdaMaxNm;
wavelengthNm = wavelengthNm(valid);
selectedX = selectedX(valid);

% El recorrido debe hacerse desde la menor hasta la mayor longitud de onda.
[wavelengthNm, order] = sort(wavelengthNm, 'ascend');
selectedX = selectedX(order);

if isempty(wavelengthNm)
    error('No hay valores validos de wavelength_nm y selected_x para graficar.');
end

%% Grafica de los datos originales
fig = figure('Color', 'w', 'Position', [100 100 1350 650]);
plot(wavelengthNm, selectedX, '-o', ...
    'Color', [0.00 0.45 0.74], ...
    'MarkerFaceColor', [0.00 0.45 0.74], ...
    'MarkerSize', 3, ...
    'LineWidth', 1.0);

xlabel('Longitud de onda (nm)');
ylabel('selected\_x (pixel)', 'Interpreter', 'tex');
title('Posicion selected\_x en funcion de la longitud de onda');
grid on;
box on;

%% Guardar la grafica
pngPath = fullfile(outputDir, '00_selected_x_vs_longitud_onda.png');
figPath = fullfile(outputDir, '00_selected_x_vs_longitud_onda.fig');

exportgraphics(fig, pngPath, 'Resolution', 180);
savefig(fig, figPath);

fprintf('\nGrafica creada correctamente.\n');
fprintf('Puntos graficados: %d\n', numel(wavelengthNm));
fprintf('Intervalo: %.1f a %.1f nm\n', wavelengthNm(1), wavelengthNm(end));
fprintf('Archivo PNG: %s\n', pngPath);
