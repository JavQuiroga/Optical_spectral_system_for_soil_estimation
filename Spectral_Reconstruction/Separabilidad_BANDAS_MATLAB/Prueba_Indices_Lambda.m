clear;
clc;
close all;

%% =========================================================
% ARCHIVOS
% ==========================================================

archivo_soil     = 'Soil_1__cube_20260617_171738_recorte.csv';
archivo_todos    = 'todos_los_puntos_con_decision.csv';
archivo_filtrado = 'bandas_sin_retrocesos_2px_hasta_1469nm.csv';

%% =========================================================
% 1. LEER LOS TRES CSV
% ==========================================================

soil = readtable(archivo_soil);
todos = readtable(archivo_todos);
filtrado = readtable(archivo_filtrado);

%% =========================================================
% 2. OBTENER LA FIRMA DE REFLECTANCIA
% ==========================================================

% No usamos band_index.
% Tomamos solamente el orden de los valores de soil_reflectance.

Rsoil = soil.soil_reflectance;

Nsoil = length(Rsoil);

fprintf('Numero de valores Soil: %d\n', Nsoil);

%% =========================================================
% 3. NUMERO DE POSICIONES DE LA FIRMA COMPLETA
% ==========================================================

% Se usa source_row como referencia real.
Ntotal = max(todos.source_row);

fprintf('Numero de posiciones de la firma completa: %d\n', Ntotal);

%% =========================================================
% 4. SOURCE_ROW QUE SOBREVIVIERON AL FILTRADO
% ==========================================================

s = filtrado.source_row;

%% =========================================================
% 5. BUSCAR EL WAVELENGTH CORRESPONDIENTE A CADA SOURCE_ROW
% ==========================================================

% Inicializamos
wavelength = nan(size(s));

for k = 1:length(s)

    idx = find(todos.source_row == s(k), 1);

    if ~isempty(idx)
        wavelength(k) = todos.wavelength_nm(idx);
    else
        warning('source_row %d no existe en todos_los_puntos.', s(k));
    end

end

%% =========================================================
% 6. CONVERTIR SOURCE_ROW A POSICION EQUIVALENTE EN SOIL
% ==========================================================

% Formula:
%
% p = 1 + (s-1)*(Nsoil-1)/(Ntotal-1)
%
% Esto garantiza:
% source_row = 1       -> posicion Soil = 1
% source_row = Ntotal  -> posicion Soil = Nsoil

p = 1 + (s - 1) .* (Nsoil - 1) ./ (Ntotal - 1);

%% =========================================================
% 7. INTERPOLAR REFLECTANCIA
% ==========================================================

% Posiciones reales de la firma Soil
x_soil = (1:Nsoil)';

% interp1:
% Si p coincide con un entero -> usa el valor original.
% Si p es decimal -> interpola linealmente.

R_interp = interp1( ...
    x_soil, ...
    Rsoil, ...
    p, ...
    'linear' ...
);

%% =========================================================
% 8. DESCARTAR SOURCE_ROW QUE NO EXISTAN EN LA TABLA COMPLETA
% ==========================================================

validos = ~isnan(wavelength) & ~isnan(R_interp);

s_valid         = s(validos);
wavelength_valid = wavelength(validos);
p_valid         = p(validos);
R_valid         = R_interp(validos);

%% =========================================================
% 9. CREAR TABLA DE RESULTADOS
% ==========================================================

resultado = table( ...
    s_valid, ...
    wavelength_valid, ...
    p_valid, ...
    R_valid, ...
    'VariableNames', { ...
        'source_row', ...
        'wavelength_nm', ...
        'posicion_equivalente_soil', ...
        'soil_reflectance_interpolada' ...
    } ...
);

disp(resultado);

%% =========================================================
% 10. GUARDAR RESULTADO
% ==========================================================

writetable(resultado, 'firma_soil_interpolada.csv');

fprintf('\nArchivo guardado: firma_soil_interpolada.csv\n');

%% =========================================================
% 11. GRAFICAR
% ==========================================================

figure;

plot( ...
    wavelength_valid, ...
    R_valid, ...
    '-o', ...
    'LineWidth', 1.2, ...
    'MarkerSize', 3 ...
);

grid on;

xlabel('Wavelength (nm)');
ylabel('Soil Reflectance');

title('Firma Soil_1 - 84 bandas Separables');
