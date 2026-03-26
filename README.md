# 🌿 Proyecto Invernadero: Modelo ML para Riego en Cascada

Este proyecto contiene un modelo de Machine Learning (Support Vector Machine - SVM) optimizado para predecir cuándo activar un sistema de riego en cascada en un invernadero, basándose en la temperatura, la humedad del aire y la humedad de la matera inferior.

## 🚀 Instrucciones de Instalación y Ejecución

Sigue estos pasos para ejecutar el proyecto en tu computadora local sin errores de dependencias.

### Paso 1: Clonar el repositorio
Abre tu terminal y clona este proyecto (o descarga el ZIP y extráelo):
\`\`\`bash
git clone <URL_DE_TU_REPOSITORIO>
cd proyecto_invernadero
\`\`\`

### Paso 2: Crear y activar el entorno virtual
Para evitar conflictos con otras librerías en tu PC, crearemos un entorno aislado. Ejecuta en tu terminal:

**En Windows:**
\`\`\`cmd

Remove-Item -Recurse -Force venv

python -m venv venv
venv\Scripts\activate
\`\`\`

**En Mac/Linux:**
\`\`\`bash
python3 -m venv venv
source venv/bin/activate
\`\`\`
*(Deberías ver `(venv)` al inicio de tu línea de comandos).*

### Paso 3: Actualizar pip e instalar requerimientos
Para evitar errores de compilación (como C++ en Windows), primero actualizamos el instalador y luego descargamos las librerías:
\`\`\`cmd
python -m pip install --upgrade pip
pip install -r requerimientos.txt
\`\`\`

### Paso 4: Generar el Dataset Simulado
Antes de entrenar el modelo, necesitamos datos. Ejecuta el generador matemático que creará un archivo CSV simulando 10,000 registros del invernadero:
\`\`\`cmd
python generar_dataset.py
\`\`\`
*(Esto creará la carpeta `datos/` y el archivo `invernadero_cascada.csv` adentro).*

### Paso 5: Entrenar el Modelo
Una vez generado el dataset, ejecuta el script principal. Este código entrenará varios modelos, optimizará el SVM, mostrará la Matriz de Confusión y guardará el modelo listo para producción.
\`\`\`cmd
python entrenar_modelo.py
\`\`\`
*(Nota: El script se pausará cuando muestre la gráfica. Cierra la ventana de la gráfica para que el código termine de ejecutarse).*

Al finalizar, verás dos archivos nuevos: `scaler_cascada.pkl` y `modelo_svm_cascada.pkl`. Estos son los archivos exportados para usarlos en producción (ej. en una Raspberry Pi).