%% ============================================================
% SELECCION MANUAL COMPLETA DE FIRMAS SOIL / WHITE / DARK
%
% Este script combina:
%   1) carga_datos.m
%   2) seleccionar_firmas_reflectancia.m
%
% Diferencias frente al script original:
%   - Carga el .npy dentro del mismo script.
%   - Guarda la firma completa desde 1:nBandas.
%   - Usa nombres equivalentes al flujo automatico:
%       soil_signature, white_signature, dark_signature, soil_reflectance
%   - Guarda resultados en:
%       <outputDir>/cubos_manuales/<cube_id>/
%   - Si MATLAB tiene acceso a Python + NumPy, tambien guarda:
%       resultado.npz
%     con las mismas llaves principales del automatico.
%
% Convencion esperada del cubo .npy:
%   y_lambda_x => cube(fila, banda, columna)
%% ============================================================

clear;
clc;
close all;

%% ------------------------------------------------------------
% 1. Parametros de entrada/salida
%% ------------------------------------------------------------

defaultCubePath = fullfile(pwd, 'Spectral_Reconstruction', 'Capturas_soil');
defaultOutputDir = fullfile(pwd, 'Spectral_Reconstruction', 'Firmas_automaticas');

[archivo, carpeta] = uigetfile({'*.npy', 'Cubos NumPy (*.npy)'}, ...
    'Seleccione el cubo .npy', defaultCubePath);

if isequal(archivo, 0)
    error('Seleccion cancelada por el usuario.');
end

ruta = fullfile(carpeta, archivo);

respuestaSalida = inputdlg( ...
    {'Carpeta de salida:'}, ...
    'Salida de firmas manuales', ...
    [1 80], ...
    {defaultOutputDir});

if isempty(respuestaSalida)
    error('Seleccion cancelada por el usuario.');
end

outputDir = respuestaSalida{1};

%% ------------------------------------------------------------
% 2. Cargar cubo .npy
%% ------------------------------------------------------------

cube = cargarNpy(ruta);

if ndims(cube) ~= 3
    error('cube debe ser un arreglo 3D con forma: filas x bandas x columnas.');
end

[nFilas, nBandas, nColumnas] = size(cube);
bandasCompletas = 1:nBandas;

fprintf('Cubo cargado: %s\n', ruta);
fprintf('Dimensiones: filas=%d, bandas=%d, columnas=%d\n', nFilas, nBandas, nColumnas);

%% ------------------------------------------------------------
% 3. Elegir banda para visualizar ROIs
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

figRois = figure('Name', 'Seleccion de ROIs manuales', 'Color', 'w');
imagesc(imgVista);
axis image;
colormap turbo;
colorbar;
title(sprintf('Banda %d - seleccione SOIL, WHITE/SPECTRALON y DARK', bandaVista));
xlabel('Columna espacial');
ylabel('Fila espacial');

%% ------------------------------------------------------------
% 4. Seleccionar ROIs manuales
%% ------------------------------------------------------------

roiSoil = seleccionarPoligono('SOIL / muestra de suelo');
maskSoil = crearMascaraDesdePoligono(roiSoil, nFilas, nColumnas);
mostrarContorno(maskSoil, 'y', 'SOIL');

roiWhite = seleccionarPoligono('WHITE / Spectralon');
maskWhite = crearMascaraDesdePoligono(roiWhite, nFilas, nColumnas);
mostrarContorno(maskWhite, 'w', 'WHITE');

roiDark = seleccionarPoligono('DARK / negro');
maskDark = crearMascaraDesdePoligono(roiDark, nFilas, nColumnas);
mostrarContorno(maskDark, 'k', 'DARK');

masks.soil = maskSoil;
masks.white = maskWhite;
masks.dark = maskDark;

rois.soil = roiSoil;
rois.white = roiWhite;
rois.dark = roiDark;

%% ------------------------------------------------------------
% 5. Calcular firmas completas 1:nBandas
%% ------------------------------------------------------------

fprintf('Calculando firmas medias completas para %d bandas...\n', nBandas);

soil_signature = calcularFirmaMedia(cube, maskSoil);
white_signature = calcularFirmaMedia(cube, maskWhite);
dark_signature = calcularFirmaMedia(cube, maskDark);

denominador = white_signature - dark_signature;
denominador(abs(denominador) < eps) = NaN;

soil_reflectance = (soil_signature - dark_signature) ./ denominador;
white_reflectance = (white_signature - dark_signature) ./ denominador;
dark_reflectance = (dark_signature - dark_signature) ./ denominador;

% Alias estilo MATLAB anterior.
firmaSoilRawCompleta = soil_signature;
firmaWhiteRawCompleta = white_signature;
firmaBlackRawCompleta = dark_signature;
reflectanciaSoilCompleta = soil_reflectance;
reflectanciaWhiteCompleta = white_reflectance;
reflectanciaBlackCompleta = dark_reflectance;

%% ------------------------------------------------------------
% 6. Crear carpeta de salida compatible con flujo automatico
%% ------------------------------------------------------------

cube_id = construirCubeId(ruta);
cubeOut = fullfile(outputDir, 'cubos_manuales', cube_id);

if ~exist(cubeOut, 'dir')
    mkdir(cubeOut);
end

archivoMat = fullfile(cubeOut, 'resultado_manual.mat');
archivoNpz = fullfile(cubeOut, 'resultado.npz');
archivoCsv = fullfile(cubeOut, 'firmas.csv');
archivoMetadata = fullfile(cubeOut, 'metadata.json');
archivoDiagnostico = fullfile(cubeOut, 'diagnostico.png');

%% ------------------------------------------------------------
% 7. Guardar .mat y CSV
%% ------------------------------------------------------------

save(archivoMat, ...
    'cube_id', ...
    'ruta', ...
    'bandasCompletas', ...
    'bandaVista', ...
    'soil_signature', ...
    'white_signature', ...
    'dark_signature', ...
    'soil_reflectance', ...
    'white_reflectance', ...
    'dark_reflectance', ...
    'firmaSoilRawCompleta', ...
    'firmaWhiteRawCompleta', ...
    'firmaBlackRawCompleta', ...
    'reflectanciaSoilCompleta', ...
    'reflectanciaWhiteCompleta', ...
    'reflectanciaBlackCompleta', ...
    'masks', ...
    'rois', ...
    '-v7.3');

tablaFirmas = table( ...
    bandasCompletas(:), ...
    soil_signature(:), ...
    white_signature(:), ...
    dark_signature(:), ...
    soil_reflectance(:), ...
    white_reflectance(:), ...
    dark_reflectance(:), ...
    'VariableNames', { ...
        'idx', ...
        'soil_signature', ...
        'white_signature', ...
        'dark_signature', ...
        'soil_reflectance', ...
        'white_reflectance', ...
        'dark_reflectance'});

writetable(tablaFirmas, archivoCsv);

npzGuardado = false;
npzError = '';
try
    preview_manual = imgVista;
    preview_range = [bandaVista, bandaVista + 1];
    preview_subranges = preview_range;
    guardarResultadoNpz(archivoNpz, ...
        soil_signature, ...
        white_signature, ...
        dark_signature, ...
        soil_reflectance, ...
        maskSoil, ...
        maskWhite, ...
        maskDark, ...
        preview_manual, ...
        preview_range, ...
        preview_subranges);
    npzGuardado = true;
catch ME
    npzError = ME.message;
    warning('No se pudo guardar resultado.npz desde MATLAB: %s', npzError);
    warning('El .mat y .csv si fueron guardados. Revise que MATLAB tenga Python + NumPy configurado.');
end

metadata = struct();
metadata.cube_id = cube_id;
metadata.input_path = ruta;
metadata.status = 'manual';
metadata.reason = '';
metadata.cube_shape_y_lambda_x = [nFilas, nBandas, nColumnas];
metadata.banda_vista = bandaVista;
metadata.bandas_guardadas = [1, nBandas];
metadata.soil_pixels = nnz(maskSoil);
metadata.white_pixels = nnz(maskWhite);
metadata.dark_pixels = nnz(maskDark);
metadata.resultado_manual_mat = archivoMat;
metadata.resultado_npz = archivoNpz;
metadata.resultado_npz_guardado = npzGuardado;
metadata.resultado_npz_error = npzError;
metadata.firmas_csv = archivoCsv;

fid = fopen(archivoMetadata, 'w');
if fid < 0
    error('No se pudo crear metadata: %s', archivoMetadata);
end
fprintf(fid, '%s', jsonencode(metadata, 'PrettyPrint', true));
fclose(fid);

%% ------------------------------------------------------------
% 8. Guardar diagnostico visual
%% ------------------------------------------------------------

figDiag = figure('Name', 'Diagnostico manual', 'Color', 'w', 'Position', [100 100 1200 850]);

subplot(2, 2, 1);
imagesc(imgVista);
axis image;
colormap turbo;
colorbar;
title(sprintf('Banda vista %d', bandaVista));
hold on;
mostrarContorno(maskSoil, 'y', 'SOIL');
mostrarContorno(maskWhite, 'w', 'WHITE');
mostrarContorno(maskDark, 'k', 'DARK');

subplot(2, 2, 2);
plot(bandasCompletas, white_signature, 'LineWidth', 1.2, 'Color', [0.05 0.05 0.05]);
hold on;
plot(bandasCompletas, soil_signature, 'LineWidth', 1.2, 'Color', [0.55 0.28 0.08]);
plot(bandasCompletas, dark_signature, 'LineWidth', 1.2, 'Color', [0.10 0.35 0.80]);
grid on;
xlabel('Indice de banda');
ylabel('Intensidad media');
title('Firmas crudas completas');
legend({'WHITE', 'SOIL', 'DARK'}, 'Location', 'best');

subplot(2, 1, 2);
plot(bandasCompletas, soil_reflectance, 'LineWidth', 1.4, 'Color', [0.55 0.28 0.08]);
grid on;
xlabel('Indice de banda');
ylabel('Reflectancia relativa');
title('Reflectancia SOIL completa');
ajustarLimiteY(soil_reflectance);

saveas(figDiag, archivoDiagnostico);

fprintf('Listo.\n');
fprintf('Firma completa guardada: 1:%d\n', nBandas);
fprintf('Carpeta de salida: %s\n', cubeOut);
fprintf('MAT: %s\n', archivoMat);
fprintf('NPZ: %s\n', archivoNpz);
fprintf('CSV: %s\n', archivoCsv);
fprintf('Metadata: %s\n', archivoMetadata);
fprintf('Diagnostico: %s\n', archivoDiagnostico);
fprintf('Pixeles ROI soil:  %d\n', nnz(maskSoil));
fprintf('Pixeles ROI white: %d\n', nnz(maskWhite));
fprintf('Pixeles ROI dark:  %d\n', nnz(maskDark));

%% ============================================================
% FUNCIONES LOCALES
%% ============================================================

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
        error('No se pudo leer correctamente el encabezado .npy.');
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
        error('Archivo incompleto. Se esperaban %d elementos y se leyeron %d.', ...
            nElements, numel(raw));
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

function cube_id = construirCubeId(ruta)
    [carpeta, nombreBase, ~] = fileparts(ruta);
    [~, nombreCarpeta] = fileparts(carpeta);

    if startsWith(nombreCarpeta, 'Soil_', 'IgnoreCase', true) && startsWith(nombreBase, 'cube_')
        cube_id = sprintf('%s__%s', nombreCarpeta, nombreBase);
    else
        cube_id = nombreBase;
    end
end

function guardarResultadoNpz(archivoNpz, soil_signature, white_signature, dark_signature, ...
    soil_reflectance, maskSoil, maskWhite, maskDark, preview_manual, preview_range, preview_subranges)

    np = py.importlib.import_module('numpy');

    soil_signature_np = np.asarray(soil_signature(:)');
    white_signature_np = np.asarray(white_signature(:)');
    dark_signature_np = np.asarray(dark_signature(:)');
    soil_reflectance_np = np.asarray(soil_reflectance(:)');
    soil_signature_np = soil_signature_np.ravel();
    white_signature_np = white_signature_np.ravel();
    dark_signature_np = dark_signature_np.ravel();
    soil_reflectance_np = soil_reflectance_np.ravel();

    soil_mask_np = np.asarray(logical(maskSoil));
    white_mask_np = np.asarray(logical(maskWhite));
    dark_mask_np = np.asarray(logical(maskDark));
    preview_np = np.asarray(single(preview_manual));

    preview_range_np = np.asarray(int32(preview_range));
    preview_range_np = preview_range_np.ravel();
    preview_subranges_np = np.asarray(int32(preview_subranges));

    np.savez_compressed(archivoNpz, pyargs( ...
        'soil_signature', soil_signature_np, ...
        'white_signature', white_signature_np, ...
        'dark_signature', dark_signature_np, ...
        'soil_reflectance', soil_reflectance_np, ...
        'soil_mask', soil_mask_np, ...
        'white_mask', white_mask_np, ...
        'dark_mask', dark_mask_np, ...
        'preview', preview_np, ...
        'preview_range', preview_range_np, ...
        'preview_subranges', preview_subranges_np));
end

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
