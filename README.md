# IntelliGest

Proyecto consolidado para clasificación de gestos con una sola aplicación de escritorio, inferencia ONNX,
entrenamiento/exportación mediante una sola copia integrada de YOLOv5 e integración UDP configurable.

Los datasets y modelos históricos están preservados localmente en `data/legacy` y `models/legacy`. Sus rutas
predeterminadas se pueden sobrescribir en `configs/paths.local.json`; ambos directorios están ignorados por Git
por su tamaño y deben respaldarse por separado.

## Requisitos e instalación

- Python 3.10 o posterior.
- Los directorios locales `data/legacy` y `models/legacy` conservados junto al proyecto.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[desktop]"
Copy-Item configs\paths.local.example.json configs\paths.local.json
intelligest profiles
intelligest paths
```

Edita `configs/paths.local.json` si los datos o modelos están en otra ubicación. Este archivo es local y está ignorado
por Git.

## Estructura

```text
configs/                  perfiles, contratos de modelo, rutas y acciones UDP
src/intelligest/app/      única aplicación PySide6 basada en CIIMA_Visio_AI
src/intelligest/inference motor ONNX y preprocesamiento
src/intelligest/training/ comandos para el YOLOv5 externo de YARVIS
src/intelligest/export/   exportación ONNX
src/intelligest/integrations/ transporte UDP
third_party/yolov5/       única implementación YOLOv5 preservada
data/legacy/              datasets locales (ignorado por Git)
models/legacy/            modelos locales (ignorado por Git)
tests/                    pruebas estáticas y unitarias sin cámara ni red
```

## Perfiles

| Perfil | Clases | Dataset predeterminado | Estado |
|---|---:|---|---|
| `ciima_4` | 4 | `data/legacy/CIIMA_Visio_AI/...` | Perfil principal |
| `intelligest_8` | 8 | `data/legacy/IntelliGest/...` | Dataset heredado |
| `yarvis_4` | 4 | `data/legacy/YARVIS/...` | Perfil heredado |
| `visio_8_legacy` | 8 | Sin ruta | Requiere `--dataset` |

El orden de clases vive en `configs/datasets/` y debe coincidir con el contrato de `configs/models/`.

## Aplicación de escritorio

La configuración se puede comprobar sin cargar ONNX, cámara ni UDP:

```powershell
intelligest-desktop --check-config --no-udp
```

Para iniciar la aplicación con el contrato `ciima_4`:

```powershell
intelligest-desktop --source 0
```

Usa `--model C:\ruta\modelo.onnx`, `--contract <archivo.json>` o `--actions <archivo.json>` para sobrescribir la
configuración. `--no-udp` desactiva el envío de acciones. También se admite una ruta de imagen o video en `--source`.

## Entrenamiento futuro

El consolidado reutiliza una sola implementación de YOLOv5 en `third_party/yolov5`, fijada en
`configs/toolchain.json`. Instala sus dependencias antes de entrenar:

```powershell
python -m pip install -r third_party\yolov5\requirements.txt
python -m pip install -e ".[train]"
intelligest train --profile ciima_4 --epochs 100 --batch-size 8 --device cpu
```

Sin `--execute`, el último comando solo imprime la invocación. Añádelo cuando quieras iniciar realmente el
entrenamiento. Puedes reemplazar el dataset con `--dataset C:\ruta\arm_poses_cls`.

## Exportación ONNX

```powershell
intelligest export-onnx --weights C:\ruta\best.pt --imgsz 224
```

El comando también es una vista previa hasta añadir `--execute`. La exportación usa `export.py` del mismo checkout
integrado de YOLOv5.

## UDP

`configs/actions/ciima_4.json` define host, puerto, broadcast, estabilidad mínima, confianza y payload por clase.
La aplicación solo envía cuando UDP está habilitado y la predicción permanece estable. Las pruebas verifican el
mapeo sin abrir sockets.

## Comprobaciones de desarrollo

```powershell
python -m compileall -q src tests
ruff check .
python -m pytest
```

Se verificaron compilación, imports seguros, JSON/TOML, ayudas CLI, configuración de la aplicación, lint y 8 pruebas
unitarias. La preservación de datasets, modelos y YOLOv5 se verificó archivo por archivo con SHA-256; consulta
`reports/preservation-verification.json`. No se ejecutaron entrenamiento, evaluación, exportación, inferencia,
cámara ni tráfico UDP; esas rutas
requieren validación posterior con dependencias, hardware, datasets y modelos reales.

## Licencia

AGPL-3.0. Consulta [LICENSE](LICENSE) y [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) para la atribución de
Ultralytics YOLOv5 y la procedencia de los repositorios históricos.
