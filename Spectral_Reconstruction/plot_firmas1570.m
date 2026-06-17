clear;
clc;
close all;

%% ============================================================
% PLOTEAR FIRMAS DE REFLECTANCIA UPSAMPLEADAS A 1570 PUNTOS
%
% Este script carga:
%   Capturas_soil/Firmas_1570/RESUMEN/todas_las_firmas_1570.mat
%
% Variables esperadas:
%   firmas1570 : matriz muestras x bandas, por ejemplo 20 x 1570
%   nombres    : nombres de las muestras
%% ============================================================

rutaMat = fullfile( ...
    'Capturas_soil', ...
    'Firmas_1570', ...
    'RESUMEN', ...
    'todas_las_firmas_1570.mat');

carpetaSalida = fullfile('Capturas_soil', 'Firmas_1570', 'PLOTS');

if ~exist(carpetaSalida, 'dir')
    mkdir(carpetaSalida);
end

if ~exist(rutaMat, 'file')
    error('No existe el archivo: %s', rutaMat);
end

data = load(rutaMat);

if ~isfield(data, 'firmas1570')
    error('El archivo no contiene la variable firmas1570.');
end

firmas = data.firmas1570;

if isfield(data, 'nombres')
    nombres = cellstr(string(data.nombres));
else
    nombres = arrayfun(@(i) sprintf('firma_%02d', i), 1:size(firmas, 1), 'UniformOutput', false);
end

if size(firmas, 2) ~= 1570 && size(firmas, 1) == 1570
    firmas = firmas.';
end

[nFirmas, nBandas] = size(firmas);
eje = (1:nBandas).';

fprintf('Firmas cargadas: %d\n', nFirmas);
fprintf('Puntos por firma: %d\n', nBandas);

%% ------------------------------------------------------------
% 1. Plot de todas las firmas
%% ------------------------------------------------------------

fig = figure('Name', 'Firmas 1570', 'Color', 'w');
plot(eje, firmas.', 'LineWidth', 1.2);
grid on;
xlabel('Indice espectral re-muestreado');
ylabel('Reflectancia');
title(sprintf('Firmas de reflectancia upsampleadas (%d x %d)', nFirmas, nBandas));

if nFirmas <= 30
    legend(nombres, 'Location', 'bestoutside', 'Interpreter', 'none');
end

guardarFigura(fig, fullfile(carpetaSalida, 'firmas1570_todas.png'));

%% ------------------------------------------------------------
% 2. Plot con media +/- desviacion estandar
%% ------------------------------------------------------------

mediaFirma = mean(firmas, 1, 'omitnan');
stdFirma = std(firmas, 0, 1, 'omitnan');

fig = figure('Name', 'Media firmas 1570', 'Color', 'w');
hold on;

x = eje(:);
y1 = (mediaFirma - stdFirma).';
y2 = (mediaFirma + stdFirma).';

fill([x; flipud(x)], [y1; flipud(y2)], [0.75 0.85 1.0], ...
    'EdgeColor', 'none', ...
    'FaceAlpha', 0.45);

plot(eje, mediaFirma, 'b', 'LineWidth', 2.0);
grid on;
xlabel('Indice espectral re-muestreado');
ylabel('Reflectancia');
title('Media de firmas 1570 +/- desviacion estandar');
legend({'media +/- std', 'media'}, 'Location', 'best');

guardarFigura(fig, fullfile(carpetaSalida, 'firmas1570_media_std.png'));

%% ------------------------------------------------------------
% 3. Plot individual por firma
%% ------------------------------------------------------------

carpetaIndividuales = fullfile(carpetaSalida, 'individuales');
if ~exist(carpetaIndividuales, 'dir')
    mkdir(carpetaIndividuales);
end

for i = 1:nFirmas
    fig = figure('Name', nombres{i}, 'Color', 'w', 'Visible', 'off');
    plot(eje, firmas(i, :), 'LineWidth', 1.8);
    grid on;
    xlabel('Indice espectral re-muestreado');
    ylabel('Reflectancia');
    title(sprintf('Firma 1570: %s', nombres{i}), 'Interpreter', 'none');

    nombreSeguro = regexprep(nombres{i}, '[^\w\-]', '_');
    guardarFigura(fig, fullfile(carpetaIndividuales, [nombreSeguro '.png']));
    close(fig);
end

fprintf('Graficas guardadas en: %s\n', carpetaSalida);

%% ============================================================
% FUNCIONES LOCALES
%% ============================================================

function guardarFigura(fig, ruta)
    try
        exportgraphics(fig, ruta, 'Resolution', 150);
    catch
        saveas(fig, ruta);
    end
end
