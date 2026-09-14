%% SEPARAR_POR_CAMBIO_CONSECUTIVO
% Conserva la primera longitud de onda y, a partir de ella, solamente los
% puntos cuyo selected_x se separa del ultimo conservado por el salto minimo.
%
% pixelStep define cuantos pixeles deben separar dos bandas.

clear;
close all;
clc;

%% Rutas
scriptPath = mfilename('fullpath');
scriptDir = fileparts(scriptPath);

inputCsv = fullfile(scriptDir, '..', ...
    'spectral_characterization_plots', 'spectral_characterization.csv');
outputDir = fullfile(scriptDir, 'resultados');
lambdaMaxNm = 1469;
pixelStep = 8;

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% Leer y validar los datos originales
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
sourceRow = (1:height(T))';

% Se excluyen filas sin valores numericos validos y longitudes superiores
% al limite acordado para el intervalo util.
valid = isfinite(wavelengthNm) & isfinite(selectedX) & ...
    wavelengthNm <= lambdaMaxNm;
wavelengthNm = wavelengthNm(valid);
selectedX = selectedX(valid);
sourceRow = sourceRow(valid);

% El analisis avanza desde la menor hasta la mayor longitud de onda.
[wavelengthNm, order] = sort(wavelengthNm, 'ascend');
selectedX = selectedX(order);
sourceRow = sourceRow(order);

if isempty(wavelengthNm)
    error('No hay valores validos para analizar.');
end

%% Excluir retrocesos sin modificar el CSV original
% Se compara cada punto contra el ultimo punto aceptado. Los valores iguales
% o mayores se admiten; los menores se descartan solamente para este analisis.
isMonotonicAccepted = false(size(selectedX));
isMonotonicAccepted(1) = true;
lastAcceptedX = selectedX(1);

for k = 2:numel(selectedX)
    if selectedX(k) >= lastAcceptedX
        isMonotonicAccepted(k) = true;
        lastAcceptedX = selectedX(k);
    end
end

monotonicWavelengthNm = wavelengthNm(isMonotonicAccepted);
monotonicSelectedX = selectedX(isMonotonicAccepted);
monotonicSourceRow = sourceRow(isMonotonicAccepted);
nNegativeRejected = sum(~isMonotonicAccepted);

%% Detectar cambios respecto al ultimo punto conservado

% La primera longitud siempre se conserva. El punto de referencia solo se
% actualiza cuando se alcanza el salto minimo configurado.
isConserved = false(size(monotonicSelectedX));
isConserved(1) = true;
lastConservedX = monotonicSelectedX(1);
deltaFromLastConserved = zeros(size(monotonicSelectedX));
deltaFromLastConserved(1) = NaN;

for k = 2:numel(monotonicSelectedX)
    deltaFromLastConserved(k) = monotonicSelectedX(k) - lastConservedX;

    if deltaFromLastConserved(k) >= pixelStep
        isConserved(k) = true;
        lastConservedX = monotonicSelectedX(k);
    end
end

conservedWavelengthNm = monotonicWavelengthNm(isConserved);
conservedSelectedX = monotonicSelectedX(isConserved);
conservedSourceRow = monotonicSourceRow(isConserved);
nSeparatedBands = numel(conservedWavelengthNm);

%% Guardar solamente las bandas finales aceptadas
separatedBands = table(conservedSourceRow, conservedWavelengthNm, ...
    conservedSelectedX, 'VariableNames', {'source_row', 'wavelength_nm', ...
    'selected_x'});

bandsCsv = fullfile(outputDir, sprintf( ...
    'bandas_sin_retrocesos_%dpx_hasta_%.0fnm.csv', ...
    pixelStep, lambdaMaxNm));

writetable(separatedBands, bandsCsv);

%% Grafica comparativa
fig = figure('Color', 'w', 'Position', [100 100 1450 700]);

plot(monotonicWavelengthNm, monotonicSelectedX, '-', ...
    'Color', [0.78 0.78 0.78], ...
    'LineWidth', 1.0, ...
    'DisplayName', 'Trayectoria sin retrocesos');
hold on;

plot(conservedWavelengthNm, conservedSelectedX, 'o-', ...
    'Color', [0.00 0.45 0.74], ...
    'MarkerFaceColor', [0.00 0.45 0.74], ...
    'MarkerSize', 3, ...
    'LineWidth', 1.0, ...
    'DisplayName', 'Puntos conservados');

xline(lambdaMaxNm, '--r', sprintf('Limite: %.0f nm', lambdaMaxNm), ...
    'LabelVerticalAlignment', 'bottom', ...
    'HandleVisibility', 'off');

xlabel('Longitud de onda (nm)');
ylabel('selected\_x (pixel)', 'Interpreter', 'tex');
title(sprintf(['Sin retrocesos | Salto minimo de %d pixeles | ' ...
    '%d bandas separables'], pixelStep, nSeparatedBands));
legend('Location', 'best');
grid on;
box on;

pngPath = fullfile(outputDir, sprintf( ...
    '03_separacion_sin_retrocesos_%dpx_hasta_%.0fnm.png', ...
    pixelStep, lambdaMaxNm));
figPath = fullfile(outputDir, sprintf( ...
    '03_separacion_sin_retrocesos_%dpx_hasta_%.0fnm.fig', ...
    pixelStep, lambdaMaxNm));

exportgraphics(fig, pngPath, 'Resolution', 180);
savefig(fig, figPath);

%% Guardar un resumen legible
summaryPath = fullfile(outputDir, sprintf( ...
    'resumen_sin_retrocesos_%dpx_hasta_%.0fnm.txt', ...
    pixelStep, lambdaMaxNm));
fid = fopen(summaryPath, 'w');
if fid == -1
    error('No se pudo crear el resumen: %s', summaryPath);
end

fprintf(fid, 'Limite maximo de longitud de onda: %.1f nm\n', lambdaMaxNm);
fprintf(fid, 'Salto minimo requerido: %d pixel(es)\n', pixelStep);
fprintf(fid, 'Puntos originales validos: %d\n', numel(wavelengthNm));
fprintf(fid, 'Puntos rechazados por retroceso: %d\n', nNegativeRejected);
fprintf(fid, 'Puntos sin retrocesos: %d\n', numel(monotonicWavelengthNm));
fprintf(fid, 'Bandas separables: %d\n', nSeparatedBands);
fprintf(fid, 'Puntos omitidos por salto insuficiente: %d\n', ...
    numel(monotonicWavelengthNm) - nSeparatedBands);
fclose(fid);

%% Resumen en la consola
fprintf('\nSeparacion terminada.\n');
fprintf('Puntos originales validos: %d\n', numel(wavelengthNm));
fprintf('Puntos rechazados por retroceso: %d\n', nNegativeRejected);
fprintf('Puntos sin retrocesos: %d\n', numel(monotonicWavelengthNm));
fprintf('Bandas separables: %d\n', nSeparatedBands);
fprintf('Salto minimo requerido: %d pixel(es)\n', pixelStep);
fprintf('Puntos omitidos por no alcanzar el salto minimo: %d\n', ...
    numel(monotonicWavelengthNm) - nSeparatedBands);
fprintf('Tabla de bandas: %s\n', bandsCsv);
fprintf('Resumen: %s\n', summaryPath);
fprintf('Resultados guardados en: %s\n', outputDir);
