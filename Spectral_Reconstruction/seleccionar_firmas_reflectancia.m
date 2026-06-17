%% ============================================================
% SELECCION INTERACTIVA DE ROIS Y CALCULO DE REFLECTANCIA
%
% Requisito:
%   Ejecutar primero carga_datos.m para tener la variable cube en workspace.
%
% Convencion usada:
%   cube(fila, banda, columna)
%   imagen espacial para seleccionar ROIs = squeeze(cube(:, bandaVista, :))
%
% Salidas en workspace:
%   firmaSoilRawRecortada, firmaWhiteRawRecortada, firmaBlackRawRecortada
%   reflectanciaSoilRecortada, reflectanciaWhiteRecortada, reflectanciaBlackRecortada
%   masks, rois
%% ============================================================

clearvars -except cube ruta dims;
clc;
close all;

if ~exist('cube', 'var')
    error('No existe la variable cube. Ejecute primero carga_datos.m.');
end

if ndims(cube) ~= 3
    error('cube debe ser un arreglo 3D con forma: filas x bandas x columnas.');
end


[nFilas, nBandas, nColumnas] = size(cube);

idxInicio = 390;
idxFin = 678;

if nBandas < idxFin
    error('El cubo solo tiene %d bandas. No se puede recortar hasta la banda %d.', nBandas, idxFin);
end

bandasRecorte = idxInicio:idxFin;

%% ------------------------------------------------------------
% 1. Elegir banda para visualizar y seleccionar areas
%% ------------------------------------------------------------

bandaVistaDefault = min(250, nBandas);
respuesta = inputdlg( ...
    {'Banda para visualizar y seleccionar ROIs:'}, ...
    'Banda de visualizacion', ...
    [1 45], ...
    {num2str(bandaVistaDefault)});

if isempty(respuesta)
    error('Seleccion cancelada por el usuario.');
end

bandaVista = round(str2double(respuesta{1}));

if isnan(bandaVista) || bandaVista < 1 || bandaVista > nBandas
    error('La banda debe estar entre 1 y %d.', nBandas);
end

imgVista = squeeze(cube(:, bandaVista, :));

figRois = figure('Name', 'Seleccion de ROIs', 'Color', 'w');
imagesc(imgVista);
axis image;
colormap turbo;
colorbar;
title(sprintf('Banda %d - seleccione SOIL, WHITE/SPECTRALON y BLACK', bandaVista));
xlabel('Columna espacial');
ylabel('Fila espacial');

%% ------------------------------------------------------------
% 2. Seleccionar ROIs interactivos
%% ------------------------------------------------------------

roiSoil = seleccionarPoligono('SOIL / muestra de suelo');
maskSoil = crearMascaraDesdePoligono(roiSoil, nFilas, nColumnas);
mostrarContorno(maskSoil, 'y', 'SOIL');

roiWhite = seleccionarPoligono('WHITE / Spectralon');
maskWhite = crearMascaraDesdePoligono(roiWhite, nFilas, nColumnas);
mostrarContorno(maskWhite, 'w', 'WHITE');

roiBlack = seleccionarPoligono('BLACK / negro');
maskBlack = crearMascaraDesdePoligono(roiBlack, nFilas, nColumnas);
mostrarContorno(maskBlack, 'k', 'BLACK');

masks.soil = maskSoil;
masks.white = maskWhite;
masks.black = maskBlack;

rois.soil = roiSoil;
rois.white = roiWhite;
rois.black = roiBlack;

%% ------------------------------------------------------------
% 3. Calcular firmas medias crudas
%% ------------------------------------------------------------

fprintf('Calculando firmas medias para %d bandas...\n', nBandas);

firmaSoilRaw = calcularFirmaMedia(cube, maskSoil);
firmaWhiteRaw = calcularFirmaMedia(cube, maskWhite);
firmaBlackRaw = calcularFirmaMedia(cube, maskBlack);

%% ------------------------------------------------------------
% 4. Recortar firmas y calcular reflectancia
%% ------------------------------------------------------------
% Modelo clasico de correccion:
%   R = (I_muestra - I_negro) / (I_blanco - I_negro)

firmaSoilRawRecortada = firmaSoilRaw(bandasRecorte);
firmaWhiteRawRecortada = firmaWhiteRaw(bandasRecorte);
firmaBlackRawRecortada = firmaBlackRaw(bandasRecorte);

denominador = firmaWhiteRawRecortada - firmaBlackRawRecortada;
denominador(abs(denominador) < eps) = NaN;

reflectanciaSoilRecortada = (firmaSoilRawRecortada - firmaBlackRawRecortada) ./ denominador;
reflectanciaWhiteRecortada = (firmaWhiteRawRecortada - firmaBlackRawRecortada) ./ denominador;
reflectanciaBlackRecortada = (firmaBlackRawRecortada - firmaBlackRawRecortada) ./ denominador;

intensidadBlancoRecortada = firmaWhiteRawRecortada;
intensidadNegroRecortada = firmaBlackRawRecortada;
intensidadSoilRecortada = firmaSoilRawRecortada;

reflectanciaBlancoRecortada = reflectanciaWhiteRecortada;
reflectanciaNegroRecortada = reflectanciaBlackRecortada;

%% ------------------------------------------------------------
% 5. Graficar firmas medias y reflectancia recortadas
%% ------------------------------------------------------------

figure('Name', 'Firmas medias crudas recortadas', 'Color', 'w');
plot(bandasRecorte, firmaWhiteRawRecortada, 'LineWidth', 1.6, 'Color', [0.05 0.05 0.05]);
hold on;
plot(bandasRecorte, firmaSoilRawRecortada, 'LineWidth', 1.6, 'Color', [0.55 0.28 0.08]);
plot(bandasRecorte, firmaBlackRawRecortada, 'LineWidth', 1.6, 'Color', [0.10 0.35 0.80]);
grid on;
xlabel('Banda espectral');
ylabel('Intensidad media');
title(sprintf('Firmas espectrales medias crudas recortadas: indices %d a %d', idxInicio, idxFin));
legend({'WHITE / Spectralon', 'SOIL', 'BLACK'}, 'Location', 'best');

figure('Name', 'Reflectancia corregida recortada', 'Color', 'w');
plot(bandasRecorte, reflectanciaSoilRecortada, 'LineWidth', 1.8, 'Color', [0.55 0.28 0.08]);
grid on;
xlabel('Banda espectral');
ylabel('Reflectancia relativa');
title(sprintf('Reflectancia SOIL corregida recortada: indices %d a %d', idxInicio, idxFin));
legend({'SOIL'}, 'Location', 'best');
ajustarLimiteY(reflectanciaSoilRecortada);

figure('Name', 'Firmas recortadas invertidas', 'Color', 'w');

subplot(2, 1, 1);
plot(bandasRecorte, fliplr(firmaWhiteRawRecortada), 'LineWidth', 1.6, 'Color', [0.05 0.05 0.05]);
hold on;
plot(bandasRecorte, fliplr(firmaSoilRawRecortada), 'LineWidth', 1.6, 'Color', [0.55 0.28 0.08]);
plot(bandasRecorte, fliplr(firmaBlackRawRecortada), 'LineWidth', 1.6, 'Color', [0.10 0.35 0.80]);
grid on;
xlabel('Banda espectral invertida visualmente');
ylabel('Intensidad media');
title('Firmas crudas recortadas invertidas');
legend({'WHITE / Spectralon', 'SOIL', 'BLACK'}, 'Location', 'best');

subplot(2, 1, 2);
plot(bandasRecorte, fliplr(reflectanciaSoilRecortada), 'LineWidth', 1.8, 'Color', [0.55 0.28 0.08]);
grid on;
xlabel('Banda espectral invertida visualmente');
ylabel('Reflectancia relativa');
title('Reflectancia SOIL recortada invertida');
legend({'SOIL'}, 'Location', 'best');
ajustarLimiteY(reflectanciaSoilRecortada);

%% ------------------------------------------------------------
% 6. Guardar firmas recortadas
%% ------------------------------------------------------------

if exist('ruta', 'var') && ~isempty(ruta)
    [carpetaSalida, nombreBase, ~] = fileparts(ruta);
else
    carpetaSalida = pwd;
    nombreBase = 'cube';
end

archivoSalida = fullfile(carpetaSalida, sprintf('%s_reflectancia_muestra_suelo.mat', nombreBase));

save(archivoSalida, ...
    'intensidadBlancoRecortada', ...
    'intensidadNegroRecortada', ...
    'intensidadSoilRecortada', ...
    'reflectanciaBlancoRecortada', ...
    'reflectanciaNegroRecortada', ...
    'reflectanciaSoilRecortada');

save(archivoSalida, 'reflectanciaSoilRecortada', 'bandasRecorte');

fprintf('Listo.\n');
fprintf('Recorte usado: indices %d a %d\n', idxInicio, idxFin);
fprintf('Archivo guardado: %s\n', archivoSalida);
fprintf('Pixeles ROI soil:  %d\n', nnz(maskSoil));
fprintf('Pixeles ROI white: %d\n', nnz(maskWhite));
fprintf('Pixeles ROI black: %d\n', nnz(maskBlack));

%% ============================================================
% FUNCIONES LOCALES
%% ============================================================

function pos = seleccionarPoligono(nombreRoi)
    title(sprintf('Dibuje ROI: %s. Doble clic para terminar.', nombreRoi));
    disp('------------------------------------------------------------');
    fprintf('Dibuje ROI: %s\n', nombreRoi);
    disp('Con drawpolygon: clics para puntos y doble clic para terminar.');
    disp('Si no aparece drawpolygon: use clics y presione ENTER para cerrar.');

    if exist('drawpolygon', 'file') == 2
        h = drawpolygon('LineWidth', 1.5);
        if isempty(h) || isempty(h.Position)
            error('ROI cancelado: %s.', nombreRoi);
        end
        pos = h.Position;
    else
        [x, y] = ginput;
        if numel(x) < 3
            error('ROI invalido: %s. Se necesitan al menos 3 puntos.', nombreRoi);
        end
        pos = [x(:), y(:)];
        hold on;
        plot([pos(:,1); pos(1,1)], [pos(:,2); pos(1,2)], 'm-', 'LineWidth', 1.5);
    end
end

function mask = crearMascaraDesdePoligono(pos, nFilas, nColumnas)
    if size(pos, 1) < 3
        error('El poligono debe tener al menos 3 vertices.');
    end

    [xGrid, yGrid] = meshgrid(1:nColumnas, 1:nFilas);
    mask = inpolygon(xGrid, yGrid, pos(:,1), pos(:,2));

    if ~any(mask(:))
        error('La mascara quedo vacia. Repita la seleccion con un area mas grande.');
    end
end

function firma = calcularFirmaMedia(cube, mask)
    [~, nBandas, ~] = size(cube);
    firma = zeros(1, nBandas);

    for k = 1:nBandas
        img = squeeze(cube(:, k, :));
        firma(k) = mean(double(img(mask)), 'omitnan');
    end
end

function mostrarContorno(mask, colorLinea, etiqueta)
    hold on;

    if exist('bwboundaries', 'file') == 2
        bordes = bwboundaries(mask);
        for i = 1:numel(bordes)
            b = bordes{i};
            plot(b(:,2), b(:,1), '-', 'Color', colorLinea, 'LineWidth', 1.5);
        end
    end

    [fila, columna] = find(mask);
    text(mean(columna), mean(fila), etiqueta, ...
        'Color', colorLinea, ...
        'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', ...
        'BackgroundColor', [0 0 0]);
end

function ajustarLimiteY(y)
    y = y(isfinite(y));

    if isempty(y)
        return;
    end

    yMin = min(y);
    yMax = max(y);
    rango = yMax - yMin;

    if rango == 0
        margen = max(abs(yMax) * 0.1, 0.01);
    else
        margen = rango * 0.10;
    end

    ylim([yMin - margen, yMax + margen]);
end
