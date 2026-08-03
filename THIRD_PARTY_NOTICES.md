# Third-party notices

## Ultralytics YOLOv5

IntelliGest conserva una sola copia de YOLOv5 en `third_party/yolov5`, extraída de `YARVIS`; su código base coincide
con [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) en el commit:

`f5ebc52c7e5130230bb8444e321244dd42c424a7`

Ese commit usa GNU Affero General Public License v3.0. Su `LICENSE`
(`be3f7b28e564e7dd05eaf59d64adba1a4065ac0e`) matches the license preserved in the historical
YARVIS. IntelliGest no incluye una segunda copia de YOLOv5.

The consolidated repository retains the same AGPL-3.0 license to avoid presenting a weaker or incompatible license
for code that integrates with the historical YOLOv5 workflow. This notice records provenance and is not legal advice.

## Historical repositories

The migration used these read-only sources owned by `LuisFelipe25`:

- IntelliGest — `c9d0653993d041a93fa5f6f0cc0194e822d74cb0`
- CIIMA_Visio_AI — `3b8df3708834568ea80ec6fcd5daf6e0d54767b6`
- Visio_AI — `d838841e00b3959c3bd034b89a21dcac12806967`
- training_model — `87dda638e80d96f16ec4e3bf5c8b94795288b9e8`
- YARVIS — `156c8ce8a9dbfcb64f1c6852eb2094ce7d3735d5`

Los datasets y modelos necesarios se preservaron en `data/legacy` y `models/legacy`, respectivamente, antes de
retirar los repositorios originales. `reports/preservation-verification.json` registra la comparación SHA-256.
