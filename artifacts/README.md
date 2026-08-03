# Modelos externos

Este repositorio no contiene pesos `.pt` ni modelos `.onnx`. Los contratos de `configs/models/` describen el orden
de clases y apuntan a los binarios que continúan en sus repositorios locales.

La aplicación acepta `--contract` y el entrenamiento/exportación aceptan rutas explícitas. Si cambian las
ubicaciones, usa `configs/paths.local.json` o modifica un contrato local sin incorporar el binario a Git.
