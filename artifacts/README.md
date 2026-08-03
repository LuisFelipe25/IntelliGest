# Modelos y Artefactos

Este directorio está destinado a almacenar artefactos generados como ejecuciones de entrenamiento (`runs/`) y binarios exportados (`.pt`, `.onnx`).

Los binarios pesados están ignorados por Git (`.gitignore`). La aplicación de escritorio acepta la ruta del modelo mediante el argumento `--model` o utiliza el contrato definido en `configs/models/arm_poses_7_app.json`.
