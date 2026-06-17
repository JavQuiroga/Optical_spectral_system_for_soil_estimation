clear;
clc;
close all;

%% ============================================================
% FALSO COLOR DESDE CUBO .NPY
%
% Soporta dos tipos de entrada:
%
% 1. Cubo reconstruido:
%       cube(y, x_scan, banda)
%    Por ejemplo:
%       preparado_muestra_colorckeher/sample_cube_y_xscan_lambda.npy
%
% 2. Cubo crudo:
%       raw(y_sensor, x_sensor, frames)
%    En ese caso se usa SLIT_Y1:SLIT_Y2 y columnas espectrales R/G/B.
%
% El resultado se guarda como PNG.
%% ============================================================

%% =========================
% CONFIGURACION
%% =========================

% Cambia esta ruta segun el cubo que quieras graficar.
rutaNpy = fullfile('preparado_muestra_colorckeher', 'sample_cube_y_xscan_lambda.npy');

% Opciones:
%   'reconstruido' -> cube(y, x_scan, banda)
%   'crudo'        -> raw(y_sensor, x_sensor, frames)
MODO_CUBO = 'reconstruido';

carpetaSalida = 'falso_color_matlab';
nombreSalida = 'falso_color.png';

% Seleccion automatica de bandas segun el numero total de bandas.
% Usa la misma idea del codigo de extraer firmas:
%   nBandas = SPECTRAL_X2 - SPECTRAL_X1
% o, si el cubo ya esta reconstruido:
%   nBandas = size(cube, 3)
%
% B = zona baja, G = zona media, R = zona alta.
AUTO_BANDS = true;
B_FRAC = 0.20;
G_FRAC = 0.50;
R_FRAC = 0.80;

% Si AUTO_BANDS = false, usa estas bandas manuales.
% En MATLAB cuentan desde 1.
R_BANDA = 200;
G_BANDA = 125;
B_BANDA = 50;

% -------------------------
% Si MODO_CUBO = 'crudo'
% -------------------------
% raw.shape = (y_sensor, x_sensor, frames)
SLIT_Y1 = 300;   % estilo Python, incluye 300
SLIT_Y2 = 850;   % estilo Python, excluye 850

% Columnas x_sensor manuales para cada canal si AUTO_BANDS = false.
R_COL = 700;
G_COL = 625;
B_COL = 550;

FLIP_SCAN_AXIS = true;
FLIP_SLIT_AXIS = false;

% Escalado visual por percentiles.
P_LOW = 1;
P_HIGH = 99;

% Gamma menor que 1 aclara sombras.
GAMMA = 0.85;

if ~exist(carpetaSalida, 'dir')
    mkdir(carpetaSalida);
end

%% =========================
% CARGAR CUBO
%% =========================

cube = cargarNpy(rutaNpy);
fprintf('Cubo cargado: %s\n', rutaNpy);
fprintf('Size: %s\n', mat2str(size(cube)));
fprintf('Clase: %s\n', class(cube));

if ndims(cube) ~= 3
    error('El archivo debe contener un arreglo 3D.');
end

%% =========================
% EXTRAER CANALES
%% =========================

switch lower(MODO_CUBO)
    case 'reconstruido'
        nBandas = size(cube, 3);

        if AUTO_BANDS
            B_BANDA = fraccionABanda(B_FRAC, nBandas);
            G_BANDA = fraccionABanda(G_FRAC, nBandas);
            R_BANDA = fraccionABanda(R_FRAC, nBandas);
        end

        validarBanda(R_BANDA, nBandas, 'R_BANDA');
        validarBanda(G_BANDA, nBandas, 'G_BANDA');
        validarBanda(B_BANDA, nBandas, 'B_BANDA');

        R = squeeze(cube(:, :, R_BANDA));
        G = squeeze(cube(:, :, G_BANDA));
        B = squeeze(cube(:, :, B_BANDA));

        etiqueta = sprintf('nBandas=%d | R=banda %d, G=banda %d, B=banda %d', nBandas, R_BANDA, G_BANDA, B_BANDA);

    case 'crudo'
        if SLIT_Y1 < 0 || SLIT_Y2 > size(cube, 1) || SLIT_Y1 >= SLIT_Y2
            error('Recorte SLIT_Y invalido.');
        end

        nBandas = SPECTRAL_X2 - SPECTRAL_X1;
        if AUTO_BANDS
            B_BANDA = fraccionABanda(B_FRAC, nBandas);
            G_BANDA = fraccionABanda(G_FRAC, nBandas);
            R_BANDA = fraccionABanda(R_FRAC, nBandas);

            B_COL = SPECTRAL_X1 + B_BANDA - 1;
            G_COL = SPECTRAL_X1 + G_BANDA - 1;
            R_COL = SPECTRAL_X1 + R_BANDA - 1;
        end

        validarColumna(R_COL, size(cube, 2), 'R_COL');
        validarColumna(G_COL, size(cube, 2), 'G_COL');
        validarColumna(B_COL, size(cube, 2), 'B_COL');

        yIdx = (SLIT_Y1 + 1):SLIT_Y2;

        R = squeeze(cube(yIdx, R_COL + 1, :));
        G = squeeze(cube(yIdx, G_COL + 1, :));
        B = squeeze(cube(yIdx, B_COL + 1, :));

        if FLIP_SLIT_AXIS
            R = flipud(R);
            G = flipud(G);
            B = flipud(B);
        end

        if FLIP_SCAN_AXIS
            R = fliplr(R);
            G = fliplr(G);
            B = fliplr(B);
        end

        etiqueta = sprintf( ...
            'nBandas=%d | R=banda %d/x %d, G=banda %d/x %d, B=banda %d/x %d', ...
            nBandas, R_BANDA, R_COL, G_BANDA, G_COL, B_BANDA, B_COL);

    otherwise
        error('MODO_CUBO debe ser reconstruido o crudo.');
end

%% =========================
% ESCALAR Y GRAFICAR
%% =========================

rgb = zeros([size(R), 3], 'single');
rgb(:, :, 1) = escalarRobusto(R, P_LOW, P_HIGH, GAMMA);
rgb(:, :, 2) = escalarRobusto(G, P_LOW, P_HIGH, GAMMA);
rgb(:, :, 3) = escalarRobusto(B, P_LOW, P_HIGH, GAMMA);

fig = figure('Name', 'Falso color', 'Color', 'w');
imshow(rgb);
axis image;
title(['Falso color - ' etiqueta], 'Interpreter', 'none');
xlabel('x\_scan / frames');
ylabel('y\_slit');

rutaSalida = fullfile(carpetaSalida, nombreSalida);
imwrite(rgb, rutaSalida);

fprintf('Falso color guardado en: %s\n', rutaSalida);

%% ============================================================
% FUNCIONES LOCALES
%% ============================================================

function validarBanda(banda, nBandas, nombre)
    if banda < 1 || banda > nBandas
        error('%s=%d fuera de rango. Debe estar entre 1 y %d.', nombre, banda, nBandas);
    end
end

function banda = fraccionABanda(frac, nBandas)
    frac = min(max(frac, 0), 1);
    banda = round(1 + frac * (nBandas - 1));
    banda = min(max(banda, 1), nBandas);
end

function validarColumna(col, ancho, nombre)
    if col < 0 || col >= ancho
        error('%s=%d fuera de rango. Debe estar entre 0 y %d.', nombre, col, ancho - 1);
    end
end

function out = escalarRobusto(img, pLow, pHigh, gammaValue)
    img = single(img);
    vals = img(isfinite(img));

    if isempty(vals)
        out = zeros(size(img), 'single');
        return;
    end

    lims = prctile(vals, [pLow pHigh]);
    vmin = lims(1);
    vmax = lims(2);

    if vmax <= vmin
        vmin = min(vals);
        vmax = max(vals);
    end

    if vmax <= vmin
        out = zeros(size(img), 'single');
        return;
    end

    out = (img - vmin) ./ (vmax - vmin);
    out = min(max(out, 0), 1);

    if gammaValue ~= 1
        out = out .^ gammaValue;
    end

    out = single(out);
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
