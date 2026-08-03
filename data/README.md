# Datasets externos

Este repositorio no contiene ni copia datasets. Los perfiles de `configs/datasets/` apuntan a las ubicaciones
actuales dentro de los cinco repositorios locales. Si se mueven esas carpetas, crea `configs/paths.local.json` a
partir del archivo de ejemplo y actualiza únicamente las rutas locales.

El código de entrenamiento recibe también `--dataset`, que prevalece sobre la ruta del perfil.
