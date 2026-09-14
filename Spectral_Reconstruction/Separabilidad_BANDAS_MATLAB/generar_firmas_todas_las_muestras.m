%% GENERAR_FIRMAS_TODAS_LAS_MUESTRAS
% Aplica las bandas sin retrocesos (2, 4, 6 y 8 px) a cada firma recortada.
% Los CSV de entrada nunca se modifican: este script solo los lee y crea
% una nueva firma interpolada para cada muestra y cada tolerancia.

clear;
clc;

%% 1. Configuracion
scriptPath = mfilename('fullpath');
scriptDir = fileparts(scriptPath);

recortesDir = fullfile(scriptDir, '..', 'Firmas_automaticas', ...
    'recortes_firmas_final', 'CSV');
todosCsv = fullfile(scriptDir, 'todos_los_puntos_con_decision.csv');
bandasDir = fullfile(scriptDir, 'resultados');
outputDir = fullfile(scriptDir, '..', 'Firmas_automaticas', ...
    'firmas_bandas_sin_retrocesos');

% Cada fila vincula la tolerancia con el CSV de longitudes seleccionado.
config = table( ...
    [2; 4; 6; 8], ...
    [84; 49; 35; 27], ...
    {'bandas_sin_retrocesos_2px_hasta_1469nm.csv'; ...
     'bandas_sin_retrocesos_4px_hasta_1469nm.csv'; ...
     'bandas_sin_retrocesos_6px_hasta_1469nm.csv'; ...
     'bandas_sin_retrocesos_8px_hasta_1469nm.csv'}, ...
    'VariableNames', {'pixel_step', 'n_bands_expected', 'bands_csv'});

if ~isfolder(recortesDir)
    error('No se encontro la carpeta de recortes: %s', recortesDir);
end
if ~isfile(todosCsv)
    error('No se encontro el CSV de referencia: %s', todosCsv);
end
if ~isfolder(bandasDir)
    error('No se encontro la carpeta de bandas: %s', bandasDir);
end
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% 2. Cargar la referencia y las cuatro selecciones de bandas
todos = readtable(todosCsv, 'VariableNamingRule', 'preserve');
requiredTodos = {'source_row', 'wavelength_nm'};
for k = 1:numel(requiredTodos)
    if ~ismember(requiredTodos{k}, todos.Properties.VariableNames)
        error('Falta la columna "%s" en %s.', requiredTodos{k}, todosCsv);
    end
end

Ntotal = max(double(todos.source_row));
if ~isfinite(Ntotal) || Ntotal < 2
    error('El numero total de posiciones debe ser al menos 2.');
end

bands = cell(height(config), 1);
for b = 1:height(config)
    bandsPath = fullfile(bandasDir, config.bands_csv{b});
    if ~isfile(bandsPath)
        error('No se encontro el CSV de bandas: %s', bandsPath);
    end

    bands{b} = readtable(bandsPath, 'VariableNamingRule', 'preserve');
    requiredBands = {'source_row', 'wavelength_nm'};
    for k = 1:numel(requiredBands)
        if ~ismember(requiredBands{k}, bands{b}.Properties.VariableNames)
            error('Falta la columna "%s" en %s.', requiredBands{k}, bandsPath);
        end
    end

    if height(bands{b}) ~= config.n_bands_expected(b)
        error(['El CSV %s contiene %d bandas, pero se esperaban %d. ' ...
            'Revise la configuracion antes de procesar.'], ...
            bandsPath, height(bands{b}), config.n_bands_expected(b));
    end

    outputSubdir = fullfile(outputDir, sprintf('%dpx_%d_bandas', ...
        config.pixel_step(b), config.n_bands_expected(b)));
    if ~exist(outputSubdir, 'dir')
        mkdir(outputSubdir);
    end
end

%% 3. Procesar todas las firmas recortadas
soilFiles = dir(fullfile(recortesDir, '*_recorte.csv'));
if isempty(soilFiles)
    error('No se encontraron archivos *_recorte.csv en %s.', recortesDir);
end

summaryRows = cell(numel(soilFiles) * height(config), 6);
summaryIndex = 0;

fprintf('Procesando %d firmas con %d tolerancias...\n', ...
    numel(soilFiles), height(config));

for f = 1:numel(soilFiles)
    inputPath = fullfile(soilFiles(f).folder, soilFiles(f).name);

    try
        soil = readtable(inputPath, 'VariableNamingRule', 'preserve');
        if ~ismember('soil_reflectance', soil.Properties.VariableNames)
            error('Falta la columna soil_reflectance.');
        end

        Rsoil = double(soil.soil_reflectance);
        Rsoil = Rsoil(isfinite(Rsoil));
        Nsoil = numel(Rsoil);
        if Nsoil < 2
            error('La firma tiene menos de dos valores validos de reflectancia.');
        end

        % Soil_1__cube_..._recorte.csv -> Soil_1
        tokens = regexp(soilFiles(f).name, '^(Soil_\d+)__', 'tokens', 'once');
        if isempty(tokens)
            error('El nombre no sigue el patron Soil_N__: %s', soilFiles(f).name);
        end
        soilId = tokens{1};

        for b = 1:height(config)
            s = double(bands{b}.source_row);
            wavelength = double(bands{b}.wavelength_nm);

            % Convierte source_row a la posicion equivalente de la firma.
            p = 1 + (s - 1) .* (Nsoil - 1) ./ (Ntotal - 1);
            Rinterp = interp1((1:Nsoil)', Rsoil, p, 'linear', NaN);

            valid = isfinite(s) & isfinite(wavelength) & isfinite(p) & ...
                isfinite(Rinterp);
            result = table(s(valid), wavelength(valid), p(valid), Rinterp(valid), ...
                'VariableNames', {'source_row', 'wavelength_nm', ...
                'posicion_equivalente_soil', 'soil_reflectance_interpolada'});

            outputSubdir = fullfile(outputDir, sprintf('%dpx_%d_bandas', ...
                config.pixel_step(b), config.n_bands_expected(b)));
            outputName = sprintf('%s__%d_Bandas.csv', soilId, ...
                config.n_bands_expected(b));
            outputPath = fullfile(outputSubdir, outputName);
            writetable(result, outputPath);

            summaryIndex = summaryIndex + 1;
            summaryRows(summaryIndex, :) = {soilFiles(f).name, soilId, ...
                config.pixel_step(b), config.n_bands_expected(b), ...
                height(result), 'ok'};
        end
    catch ME
        % Si una firma falla, se registra el error y el resto continua.
        for b = 1:height(config)
            summaryIndex = summaryIndex + 1;
            summaryRows(summaryIndex, :) = {soilFiles(f).name, '', ...
                config.pixel_step(b), config.n_bands_expected(b), ...
                0, ME.message};
        end
    end

    if mod(f, 50) == 0 || f == numel(soilFiles)
        fprintf('Procesadas: %d de %d firmas.\n', f, numel(soilFiles));
    end
end

%% 4. Guardar el resumen del lote
summaryRows = summaryRows(1:summaryIndex, :);
summary = cell2table(summaryRows, 'VariableNames', {'input_file', ...
    'soil_id', 'pixel_step', 'n_bands_expected', 'n_bands_written', 'status'});

summaryPath = fullfile(outputDir, 'resumen_procesamiento_bandas.csv');
writetable(summary, summaryPath);

nOk = sum(strcmp(summary.status, 'ok'));
nError = height(summary) - nOk;
fprintf('\nProceso terminado.\n');
fprintf('Firmas de entrada: %d\n', numel(soilFiles));
fprintf('CSV creados correctamente: %d\n', nOk);
fprintf('Errores registrados: %d\n', nError);
fprintf('Resultados: %s\n', outputDir);
