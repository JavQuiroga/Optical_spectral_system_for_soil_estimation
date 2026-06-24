# Automatización de firmas espectrales

El script `automatizar_firmas_cubos.py` procesa recursivamente cubos `.npy` y
extrae las firmas de `SOIL`, `WHITE` y `DARK` sin selección manual.

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

## 4. Revisar resultados

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
