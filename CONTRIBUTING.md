# Guía de Contribución a IntelliGest

¡Gracias por tu interés en contribuir a **IntelliGest**! Este documento proporciona pautas para colaborar en el desarrollo del sistema de clasificación de gestos y poses en tiempo real.

---

## 🛠️ Entorno de Desarrollo

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/tu-usuario/IntelliGest.git
   cd IntelliGest
   ```

2. **Crear y activar un entorno virtual**:
   - En Linux/macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - En Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```

3. **Instalar dependencias en modo editable**:
   ```bash
   python -m pip install -e ".[dev,desktop,train]"
   ```

---

## 🧪 Verificación y Pruebas

Antes de enviar un Pull Request (PR), asegúrate de que todas las pruebas pasen localmente:

```bash
# Verificación de sintaxis
python -m compileall -q src tests

# Verificación de formateo y estilo
ruff check .

# Suite de pruebas unitarias
pytest
```

---

## 📐 Convenciones del Proyecto

- **Estructura del código**: Todo el código principal debe vivir bajo `src/intelligest/`.
- **Configuraciones**: Las taxonomías de datos y contratos de modelos se definen mediante JSON en `configs/`.
- **Mensajes de Commit**: Usa commits concisos e informativos (ej. `feat: agregar perfil arm_poses_7`, `fix: corregir manejo de reconexión UDP`).

---

## 📄 Licencia

Al contribuir a IntelliGest, aceptas que tus contribuciones se licencien bajo los términos de la licencia [AGPL-3.0](LICENSE).
