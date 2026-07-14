# Automatización de firmas espectrales

El script `automatizar_firmas_cubos.py` procesa recursivamente cubos `.npy` y
extrae las firmas de `SOIL`, `WHITE` y `DARK` sin selección manual.

Las ROIs finales tienen geometría estable:

- `SOIL`: círculo interior al objeto detectado.
- `WHITE`: círculo interior al objeto detectado.
- `DARK`: rectángulo pequeño ubicado automáticamente en la zona más oscura
  de las cuatro esquinas.

## 1. Probar el algoritmo

Desde `Spectral_Reconstruction`:

```powershell
python .\automatizar_firmas_cubos.py --self-test
```

La prueba crea un cubo sintético y guarda sus diagnósticos en
`resultados_firmas_automaticas`.

## 2. Probar primero 20 cubos reales

Para cubos con forma NumPy `(fila, banda, columna)`:

```powershell
python .\automatizar_firmas_cubos.py `
  --input-dir "D:\Capturas_soil" `
  --output-dir ".\resultados_prueba_20" `
  --layout y_lambda_x `
  --preview-start 390 `
  --preview-stop 679 `
  --k-values 4,5,6,7,8 `
  --limit 20
```

`--preview-stop` es exclusivo. El intervalo anterior utiliza las bandas
390 a 678 en índices base cero.

Para cubos reconstruidos con forma `(fila, columna, banda)`, usar:

```powershell
--layout y_x_lambda
```

## 3. Añadir posiciones aproximadas

Si los objetos conservan una ubicación general, se pueden indicar sus centros
normalizados como `x,y`. Esto mejora bastante la clasificación sin imponer
coordenadas rígidas:

```powershell
--expected-soil 0.50,0.65 `
--expected-white 0.25,0.30 `
--expected-dark 0.75,0.30
```

La posición esperada de `DARK` ya no es necesaria porque se busca
automáticamente en las esquinas.

## 4. Ajustar el tamaño de las ROIs

```powershell
--circle-radius-fraction 0.22 `
--soil-radius-pixels 90 `
--dark-search-fraction 0.30 `
--dark-height-fraction 0.08 `
--dark-width-fraction 0.08
```

Las fracciones se expresan respecto al tamaño del objeto o de la imagen. Por
ejemplo, el rectángulo oscuro tendrá por defecto una altura y anchura cercanas
al 8 % de la imagen. La ROI de `SOIL` usa un radio fijo en píxeles, indicado
por `--soil-radius-pixels`.

## 5. Ajustar la segmentación

Por defecto el script ya no usa un único `K`. Prueba varias segmentaciones y
elige la que produce objetos más compactos y coherentes para `SOIL` y `WHITE`:

```powershell
--k-values 4,5,6,7,8
```

Para comparar con un único valor fijo:

```powershell
--clusters 6
```

En cada `metadata.json` queda guardado qué valores de `K` se probaron y cuál
fue seleccionado.

Además, para cada cubo se guardan resultados separados por K:

```text
cubos/Soil_X__cube_.../
├── resultado.npz
├── metadata.json
├── firmas_reflectancia_por_k.png
└── por_k/
    ├── k_04/
    ├── k_05/
    ├── k_06/
    ├── k_07/
    └── k_08/
```

Cada carpeta `k_XX` incluye su `resultado_k_XX.npz`, `metadata_k_XX.json`,
diagnóstico de segmentación y gráfica de reflectancia.

## 6. Revisar resultados

Cada cubo produce:

- `resultado.npz`: firmas, reflectancia y máscaras.
- `metadata.json`: puntuaciones y parámetros.
- `diagnostico.png`: preview, segmentación y ROIs detectadas.
- `diagnostico_reflectancia.png`: firma resultante.

En la raíz de salida se generan:

- `resumen.csv`: estado `ok` o `review` por cubo.
- `errores.json`: cubos que no pudieron procesarse.
- `todas_las_reflectancias.npy`: matriz muestras por bandas.
- `ids_reflectancias.txt`: identificador correspondiente a cada fila.

Antes de ejecutar los 989 cubos se deben revisar los diagnósticos de una muestra
variada y ajustar el rango espectral, número de grupos y posiciones esperadas.
