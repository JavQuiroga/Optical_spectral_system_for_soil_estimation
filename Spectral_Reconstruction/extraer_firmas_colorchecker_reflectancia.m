clear;
clc;
close all;

%% ============================================================
% EXTRAER FIRMAS DE REFLECTANCIA DEL COLORCHECKER
%
% Entradas esperadas:
%   Cubos .npy crudos con forma NumPy:
%       raw(y_sensor, x_sensor, frames)
%
% Flujo:
%   1. Carga cubo ColorChecker, Spectralon y negro.
%   2. Muestra una imagen espacial de cada cubo para seleccionar ROIs.
%   3. Calcula firmas crudas por banda espectral x.
%   4. Corrige reflectancia:
%       R = (I_patch - I_negro) / (I_blanco - I_negro)
%
% Nota:
%   Los cubos NO necesitan tener el mismo numero de frames.
%   Cada ROI se selecciona en su propio cubo.
%% ============================================================

%% =========================
% CONFIGURACION
%% =========================

rutaColorChecker = fullfile('Colorckeher', 'Cube_colorcheker1.npy');
rutaWhite        = fullfile('Colorckeher', 'Cube_blancoSpectralon.npy');
rutaBlack        = fullfile('Colorckeher', 'Cube_negro.npy');

carpetaSalida = 'firmas_colorchecker_reflectancia_matlab';

% raw.shape = (y_sensor, x_sensor, frames)
SLIT_Y1 = 300;
SLIT_Y2 = 850;

SPECTRAL_X1 = 500;
SPECTRAL_X2 = 750;

% MATLAB usa indices desde 1. Estos valores vienen de coordenadas tipo Python.
% Por eso se convierten internamente a indices MATLAB con +1.

FLIP_SCAN_AXIS = true;
FLIP_SLIT_AXIS = false;
FLIP_LAMBDA_AXIS = false;

% Vista para seleccionar ROIs:
%   'mean'   = promedio entre PREVIEW_X1:PREVIEW_X2
%   'max'    = maximo entre PREVIEW_X1:PREVIEW_X2
%   'single' = una sola columna PREVIEW_COL
PREVIEW_MODE = 'single';
PREVIEW_X1 = 650;
PREVIEW_X2 = 800;
PREVIEW_COL = 700;

% Numero de parches del ColorChecker a seleccionar.
% Para probar rapido use 3 o 6; para ColorChecker clasico use 24.
NUM_PATCHES = 2;

% Para rotular eje aproximado. Si ya tienes wavelengths calibradas,
% puedes reemplazar esto despues.
START_NM = 400;
END_NM = 1700;

REDUCCION_FIRMA = 'median';  % 'median' o 'mean'
DEN_EPS = 1e-6;

if ~exist(carpetaSalida, 'dir')
    mkdir(carpetaSalida);
end

%% =========================
% CARGA DE CUBOS
%% =========================

disp('Cargando cubos .npy...');
rawColor = cargarNpy(rutaColorChecker);
rawWhite = cargarNpy(rutaWhite);
rawBlack = cargarNpy(rutaBlack);

validarCuboRaw(rawColor, 'ColorChecker', SLIT_Y1, SLIT_Y2, SPECTRAL_X1, SPECTRAL_X2);
validarCuboRaw(rawWhite, 'Spectralon', SLIT_Y1, SLIT_Y2, SPECTRAL_X1, SPECTRAL_X2);
validarCuboRaw(rawBlack, 'Negro', SLIT_Y1, SLIT_Y2, SPECTRAL_X1, SPECTRAL_X2);

fprintf('ColorChecker: %s\n', mat2str(size(rawColor)));
fprintf('Spectralon:   %s\n', mat2str(size(rawWhite)));
fprintf('Negro:        %s\n', mat2str(size(rawBlack)));

if size(rawColor, 3) ~= size(rawWhite, 3) || size(rawColor, 3) ~= size(rawBlack, 3)
    disp('Nota: los cubos tienen diferente numero de frames. Esto es valido.');
    fprintf('  ColorChecker frames: %d\n', size(rawColor, 3));
    fprintf('  Spectralon frames:   %d\n', size(rawWhite, 3));
    fprintf('  Negro frames:        %d\n', size(rawBlack, 3));
end

%% =========================
% SELECCION ROI BLANCO
%% =========================

disp('Seleccione ROI del blanco / Spectralon.');
previewWhite = crearPreview(rawWhite, SLIT_Y1, SLIT_Y2, PREVIEW_MODE, PREVIEW_X1, PREVIEW_X2, PREVIEW_COL, FLIP_SCAN_AXIS, FLIP_SLIT_AXIS);
[maskWhite, roiWhite] = seleccionarRoiEnPreview(previewWhite, 'ROI blanco / Spectralon', 'w');
maskWhiteRaw = deshacerOrientacionMascara(maskWhite, FLIP_SCAN_AXIS, FLIP_SLIT_AXIS);
firmaWhite = calcularFirmaRaw(rawWhite, maskWhiteRaw, SLIT_Y1, SLIT_Y2, SPECTRAL_X1, SPECTRAL_X2, FLIP_LAMBDA_AXIS, REDUCCION_FIRMA);
fprintf('Pixeles ROI blanco: %d\n', nnz(maskWhiteRaw));

%% =========================
% SELECCION ROI NEGRO
%% =========================

disp('Seleccione ROI del negro.');
previewBlack = crearPreview(rawBlack, SLIT_Y1, SLIT_Y2, PREVIEW_MODE, PREVIEW_X1, PREVIEW_X2, PREVIEW_COL, FLIP_SCAN_AXIS, FLIP_SLIT_AXIS);
[maskBlack, roiBlack] = seleccionarRoiEnPreview(previewBlack, 'ROI negro', 'k');
maskBlackRaw = deshacerOrientacionMascara(maskBlack, FLIP_SCAN_AXIS, FLIP_SLIT_AXIS);
firmaBlack = calcularFirmaRaw(rawBlack, maskBlackRaw, SLIT_Y1, SLIT_Y2, SPECTRAL_X1, SPECTRAL_X2, FLIP_LAMBDA_AXIS, REDUCCION_FIRMA);
fprintf('Pixeles ROI negro: %d\n', nnz(maskBlackRaw));

%% =========================
% SELECCION PATCHES COLORCHECKER
%% =========================

previewColor = crearPreview(rawColor, SLIT_Y1, SLIT_Y2, PREVIEW_MODE, PREVIEW_X1, PREVIEW_X2, PREVIEW_COL, FLIP_SCAN_AXIS, FLIP_SLIT_AXIS);

nBandas = SPECTRAL_X2 - SPECTRAL_X1;
firmasRaw = zeros(NUM_PATCHES, nBandas);
reflectancias = zeros(NUM_PATCHES, nBandas);
roiPatches = cell(NUM_PATCHES, 1);
pixelesPatch = zeros(NUM_PATCHES, 1);

denominador = firmaWhite - firmaBlack;
denominador(abs(denominador) < DEN_EPS) = NaN;

for p = 1:NUM_PATCHES
    nombrePatch = sprintf('patch_%02d', p);
    fprintf('Seleccione ROI del ColorChecker: %s\n', nombrePatch);

    [maskPatch, roiPatch] = seleccionarRoiEnPreview(previewColor, ['ROI ColorChecker: ' nombrePatch], 'y');
    maskPatchRaw = deshacerOrientacionMascara(maskPatch, FLIP_SCAN_AXIS, FLIP_SLIT_AXIS);

    firmaPatch = calcularFirmaRaw(rawColor, maskPatchRaw, SLIT_Y1, SLIT_Y2, SPECTRAL_X1, SPECTRAL_X2, FLIP_LAMBDA_AXIS, REDUCCION_FIRMA);
    reflPatch = (firmaPatch - firmaBlack) ./ denominador;

    firmasRaw(p, :) = firmaPatch;
    reflectancias(p, :) = reflPatch;
    roiPatches{p} = roiPatch;
    pixelesPatch(p) = nnz(maskPatchRaw);

    fprintf('Pixeles ROI %s: %d\n', nombrePatch, pixelesPatch(p));
end

%% =========================
% GUARDAR RESULTADOS
%% =========================

sensorX = SPECTRAL_X1:(SPECTRAL_X2 - 1);
if FLIP_LAMBDA_AXIS
    sensorX = fliplr(sensorX);
end

wavelengthsNmApprox = linspace(START_NM, END_NM, nBandas);

archivoMat = fullfile(carpetaSalida, 'firmas_colorchecker_reflectancia.mat');
save(archivoMat, ...
    'firmaWhite', ...
    'firmaBlack', ...
    'firmasRaw', ...
    'reflectancias', ...
    'sensorX', ...
    'wavelengthsNmApprox', ...
    'roiWhite', ...
    'roiBlack', ...
    'roiPatches', ...
    'pixelesPatch', ...
    'rutaColorChecker', ...
    'rutaWhite', ...
    'rutaBlack', ...
    'SLIT_Y1', 'SLIT_Y2', 'SPECTRAL_X1', 'SPECTRAL_X2', ...
    'PREVIEW_MODE', 'PREVIEW_X1', 'PREVIEW_X2', 'PREVIEW_COL', ...
    'REDUCCION_FIRMA');

archivoCsv = fullfile(carpetaSalida, 'firmas_colorchecker_reflectancia.csv');
guardarCsvFirmas(archivoCsv, sensorX, wavelengthsNmApprox, firmaWhite, firmaBlack, firmasRaw, reflectancias);

graficarFirmas(carpetaSalida, wavelengthsNmApprox, firmaWhite, firmaBlack, firmasRaw, reflectancias);

fprintf('\nListo.\n');
fprintf('Archivo MAT: %s\n', archivoMat);
fprintf('Archivo CSV: %s\n', archivoCsv);
fprintf('Carpeta salida: %s\n', carpetaSalida);

%% ============================================================
% FUNCIONES LOCALES
%% ============================================================

function validarCuboRaw(raw, nombre, y1, y2, x1, x2)
    if ndims(raw) ~= 3
        error('%s debe ser 3D (y_sensor, x_sensor, frames).', nombre);
    end

    if y1 < 0 || y2 > size(raw, 1) || y1 >= y2
        error('Recorte y invalido para %s.', nombre);
    end

    if x1 < 0 || x2 > size(raw, 2) || x1 >= x2
        error('Recorte espectral x invalido para %s.', nombre);
    end
end

function preview = crearPreview(raw, y1, y2, modo, px1, px2, pcol, flipScan, flipSlit)
    yIdx = (y1 + 1):y2;

    switch lower(modo)
        case 'single'
            xIdx = pcol + 1;
            preview = squeeze(raw(yIdx, xIdx, :));

        case 'mean'
            xIdx = (px1 + 1):px2;
            preview = squeeze(mean(raw(yIdx, xIdx, :), 2));

        case 'max'
            xIdx = (px1 + 1):px2;
            preview = squeeze(max(raw(yIdx, xIdx, :), [], 2));

        otherwise
            error('PREVIEW_MODE debe ser mean, max o single.');
    end

    preview = single(preview);

    if flipSlit
        preview = flipud(preview);
    end
    if flipScan
        preview = fliplr(preview);
    end
end

function [mask, pos] = seleccionarRoiEnPreview(img, titulo, colorLinea)
    fig = figure('Name', titulo, 'Color', 'w');
    imagesc(img);
    axis image;
    colormap turbo;
    colorbar;
    title([titulo newline 'Dibuje poligono y doble clic/ENTER para terminar']);
    xlabel('x\_scan / frames');
    ylabel('y\_slit');

    if exist('drawpolygon', 'file') == 2
        h = drawpolygon('LineWidth', 1.5, 'Color', colorLinea);
        if isempty(h) || isempty(h.Position)
            close(fig);
            error('ROI cancelada.');
        end
        pos = h.Position;
    else
        [x, y] = ginput;
        if numel(x) < 3
            close(fig);
            error('ROI invalida. Se necesitan al menos 3 puntos.');
        end
        pos = [x(:), y(:)];
        hold on;
        plot([pos(:,1); pos(1,1)], [pos(:,2); pos(1,2)], '-', 'Color', colorLinea, 'LineWidth', 1.5);
    end

    [xGrid, yGrid] = meshgrid(1:size(img, 2), 1:size(img, 1));
    mask = inpolygon(xGrid, yGrid, pos(:,1), pos(:,2));

    if ~any(mask(:))
        close(fig);
        error('La mascara quedo vacia.');
    end

    hold on;
    plot([pos(:,1); pos(1,1)], [pos(:,2); pos(1,2)], '-', 'Color', colorLinea, 'LineWidth', 1.5);
    pause(0.25);
    close(fig);
end

function maskRaw = deshacerOrientacionMascara(mask, flipScan, flipSlit)
    maskRaw = mask;
    if flipScan
        maskRaw = fliplr(maskRaw);
    end
    if flipSlit
        maskRaw = flipud(maskRaw);
    end
end

function firma = calcularFirmaRaw(raw, maskRaw, y1, y2, x1, x2, flipLambda, reduccion)
    yIdx = (y1 + 1):y2;
    sensorX = (x1 + 1):x2;
    nBandas = numel(sensorX);
    firma = zeros(1, nBandas);

    if size(maskRaw, 1) ~= numel(yIdx) || size(maskRaw, 2) ~= size(raw, 3)
        error('La mascara no coincide con (y_crop, frames) del cubo.');
    end

    for k = 1:nBandas
        img = squeeze(raw(yIdx, sensorX(k), :));
        valores = double(img(maskRaw));

        switch lower(reduccion)
            case 'median'
                firma(k) = median(valores, 'omitnan');
            case 'mean'
                firma(k) = mean(valores, 'omitnan');
            otherwise
                error('REDUCCION_FIRMA debe ser median o mean.');
        end
    end

    if flipLambda
        firma = fliplr(firma);
    end
end

function guardarCsvFirmas(rutaCsv, sensorX, wavelengths, firmaWhite, firmaBlack, firmasRaw, reflectancias)
    nPatches = size(firmasRaw, 1);
    nBandas = numel(sensorX);

    fid = fopen(rutaCsv, 'w');
    if fid < 0
        error('No se pudo crear CSV: %s', rutaCsv);
    end

    fprintf(fid, 'band_index,sensor_x,wavelength_nm,white_raw,black_raw');
    for p = 1:nPatches
        fprintf(fid, ',patch_%02d_raw,patch_%02d_reflectance', p, p);
    end
    fprintf(fid, '\n');

    for k = 1:nBandas
        fprintf(fid, '%d,%d,%.8g,%.8g,%.8g', k, sensorX(k), wavelengths(k), firmaWhite(k), firmaBlack(k));
        for p = 1:nPatches
            fprintf(fid, ',%.8g,%.8g', firmasRaw(p, k), reflectancias(p, k));
        end
        fprintf(fid, '\n');
    end

    fclose(fid);
end

function graficarFirmas(carpetaSalida, wavelengths, firmaWhite, firmaBlack, firmasRaw, reflectancias)
    fig = figure('Name', 'Firmas crudas ColorChecker', 'Color', 'w');
    plot(wavelengths, firmaWhite, 'k', 'LineWidth', 1.8);
    hold on;
    plot(wavelengths, firmaBlack, 'b', 'LineWidth', 1.8);
    for p = 1:size(firmasRaw, 1)
        plot(wavelengths, firmasRaw(p, :), 'LineWidth', 1.0);
    end
    grid on;
    xlabel('Longitud de onda aprox. (nm)');
    ylabel('Intensidad');
    title('Firmas crudas');
    saveas(fig, fullfile(carpetaSalida, 'firmas_crudas_colorchecker.png'));

    fig = figure('Name', 'Reflectancia ColorChecker', 'Color', 'w');
    plot(wavelengths, reflectancias', 'LineWidth', 1.2);
    grid on;
    xlabel('Longitud de onda aprox. (nm)');
    ylabel('(I\_patch - I\_negro) / (I\_blanco - I\_negro)');
    title('Reflectancia ColorChecker');
    saveas(fig, fullfile(carpetaSalida, 'reflectancia_colorchecker.png'));
end

function cube = cargarNpy(ruta)
    fid = fopen(ruta, 'r', 'ieee-le');
    if fid < 0
        error('No se pudo abrir el archivo: %s', ruta);
    end

    magic = fread(fid, 6, '*uint8')';
    expected = uint8([147, double('NUMPY')]);

    if ~isequal(magic, expected)
        fclose(fid);
        error('El archivo no parece ser un .npy valido.');
    end

    version = fread(fid, 2, '*uint8')';

    if version(1) == 1
        headerLen = fread(fid, 1, 'uint16');
    elseif version(1) == 2 || version(1) == 3
        headerLen = fread(fid, 1, 'uint32');
    else
        fclose(fid);
        error('Version .npy no soportada.');
    end

    header = fread(fid, headerLen, '*char')';
    dataOffset = ftell(fid);
    fclose(fid);

    descrTok = regexp(header, '[''"]descr[''"]\s*:\s*[''"]([^''"]+)[''"]', 'tokens', 'once');
    fortTok  = regexp(header, '[''"]fortran_order[''"]\s*:\s*(True|False)', 'tokens', 'once');
    shapeTok = regexp(header, '[''"]shape[''"]\s*:\s*\(([^\)]*)\)', 'tokens', 'once');

    if isempty(descrTok) || isempty(fortTok) || isempty(shapeTok)
        error('No se pudo leer completamente el encabezado .npy.');
    end

    descr = descrTok{1};
    fortranOrder = strcmp(fortTok{1}, 'True');

    dimStr = regexp(shapeTok{1}, '\d+', 'match');
    dims = str2double(dimStr);

    endianChar = descr(1);
    kindChar = descr(2);
    nBytes = str2double(descr(3:end));

    switch kindChar
        case 'u'
            precision = sprintf('uint%d', nBytes * 8);
        case 'i'
            precision = sprintf('int%d', nBytes * 8);
        case 'f'
            if nBytes == 4
                precision = 'single';
            elseif nBytes == 8
                precision = 'double';
            else
                error('Tipo float de %d bytes no soportado.', nBytes);
            end
        case 'b'
            precision = 'uint8';
        otherwise
            error('Tipo de dato no soportado: %s', descr);
    end

    if endianChar == '>'
        machineFmt = 'ieee-be';
    else
        machineFmt = 'ieee-le';
    end

    fid = fopen(ruta, 'r', machineFmt);
    if fid < 0
        error('No se pudo reabrir el archivo: %s', ruta);
    end

    fseek(fid, dataOffset, 'bof');
    nElements = prod(dims);
    raw = fread(fid, nElements, ['*' precision]);
    fclose(fid);

    if numel(raw) ~= nElements
        error('Archivo incompleto. Se esperaban %d elementos y se leyeron %d.', nElements, numel(raw));
    end

    if fortranOrder
        cube = reshape(raw, dims);
    else
        cube = reshape(raw, fliplr(dims));
        cube = permute(cube, length(dims):-1:1);
    end

    if kindChar == 'b'
        cube = logical(cube);
    end
end
