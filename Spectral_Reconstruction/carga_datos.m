clear;
clc;
close all;

%% ============================================================
% CARGAR ARCHIVO .NPY EN MATLAB
% Archivo:
% C:\Users\PAGT2\Downloads\spectralon_cube.npy
%
% Resultado final:
% cube = cubo cargado desde NumPy
%
% Convencion esperada:
% cube(a,b,c)
% a = espacio vertical
% b = longitud de onda
% c = espacio horizontal
%% ============================================================

ruta = 'C:\Users\ASUS\Documents\STSIVA_TESIS\Spectral\Capturas_soil\Soil_989\cube_20260509_085042.npy';

%% ------------------------------------------------------------
% 1. Abrir archivo y leer encabezado .npy
%% ------------------------------------------------------------

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

%% ------------------------------------------------------------
% 2. Extraer informacion del encabezado
%% ------------------------------------------------------------

descrTok = regexp(header, '[''"]descr[''"]\s*:\s*[''"]([^''"]+)[''"]', 'tokens', 'once');
fortTok  = regexp(header, '[''"]fortran_order[''"]\s*:\s*(True|False)', 'tokens', 'once');
shapeTok = regexp(header, '[''"]shape[''"]\s*:\s*\(([^\)]*)\)', 'tokens', 'once');

if isempty(descrTok)
    error('No se pudo leer el tipo de dato descr del archivo .npy.');
end

if isempty(fortTok)
    error('No se pudo leer fortran_order del archivo .npy.');
end

if isempty(shapeTok)
    error('No se pudo leer shape del archivo .npy.');
end

descr = descrTok{1};
fortranOrder = strcmp(fortTok{1}, 'True');

dimStr = regexp(shapeTok{1}, '\d+', 'match');
dims = str2double(dimStr);

if isempty(dims)
    error('No se pudieron leer las dimensiones del archivo .npy.');
end

%% ------------------------------------------------------------
% 3. Interpretar tipo de dato de NumPy
%% ------------------------------------------------------------

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

%% ------------------------------------------------------------
% 4. Leer datos binarios
%% ------------------------------------------------------------

fid = fopen(ruta, 'r', machineFmt);

if fid < 0
    error('No se pudo reabrir el archivo: %s', ruta);
end

fseek(fid, dataOffset, 'bof');

nElements = prod(dims);

raw = fread(fid, nElements, ['*' precision]);

fclose(fid);

if numel(raw) ~= nElements
    error('El archivo parece incompleto. Se esperaban %d elementos, pero se leyeron %d.', ...
          nElements, numel(raw));
end

%% ------------------------------------------------------------
% 5. Reorganizar como matriz/cubo de MATLAB
%% ------------------------------------------------------------

if fortranOrder
    cube = reshape(raw, dims);
else
    cube = reshape(raw, fliplr(dims));
    cube = permute(cube, length(dims):-1:1);
end

if kindChar == 'b'
    cube = logical(cube);
end

clear raw;

%% ------------------------------------------------------------
% 6. Mostrar informacion del cubo cargado
%% ------------------------------------------------------------

disp('============================================');
disp('ARCHIVO .NPY CARGADO CORRECTAMENTE');
disp('============================================');

disp('Ruta:');
disp(ruta);

disp('Header .npy:');
disp(header);

disp('Tipo NumPy descr:');
disp(descr);

disp('Precision MATLAB:');
disp(precision);

disp('Fortran order:');
disp(fortranOrder);

disp('Dimensiones segun NumPy:');
disp(dims);

disp('Size final en MATLAB:');
disp(size(cube));

disp('Numero de dimensiones:');
disp(ndims(cube));

disp('Clase del dato:');
disp(class(cube));

disp('Valor minimo:');
disp(min(cube(:)));

disp('Valor maximo:');
disp(max(cube(:)));

disp('Variable cargada en workspace: cube');

%% ------------------------------------------------------------
% 7. Crear cubo normalizado opcional
%% ------------------------------------------------------------
% OJO:
% Esto convierte a double. Si el cubo es muy grande, puede consumir mucha RAM.
% Si no lo necesita, deje estas lineas comentadas.

% cubonorm = double(cube) ./ double(max(cube(:)));
% disp('Variable normalizada creada: cubonorm');