# COG Protein FineTuner

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ollama](https://img.shields.io/badge/Ollama-Required-green.svg)](https://ollama.ai/)

Un pipeline completo para predicción de funciones de proteínas usando categorías COG (Clusters of Orthologous Groups) con modelos de lenguaje fine-tuneados.

## 🚀 Características Principales

- **Procesamiento de datos COG** con regiones funcionales (footprints)
- **Fine-tuning** con técnicas LoRA y QLoRA
- **Prompts zero-shot y one-shot** automáticos
- **Evaluación completa** con métricas detalladas
- **Grid search** automático para optimización de hiperparámetros
- **Barras de progreso** para todas las operaciones largas
- **Compatibilidad total** entre todos los componentes

## 📦 Componentes

### 1. `cog_processor_fixed.py` - Procesador de Datos COG
Genera datasets de entrenamiento y test a partir de datos COG raw.

**Características:**
- Extracción de regiones funcionales usando footprints
- Filtros de calidad configurables
- Balanceo automático de categorías
- Generación de splits train/test estratificados

### 2. `cog_finetuner_compact.py` - Fine-tuner Compacto
Entrena modelos de lenguaje para predicción COG con técnicas avanzadas.

**Características:**
- Soporte LoRA (16-bit) y QLoRA (4-bit)
- Prompts zero-shot y one-shot automáticos
- Grid search para optimización de hiperparámetros
- Entrenamiento dual (múltiples técnicas simultáneas)
- Exportación automática a Ollama

### 3. `cog_evaluator_compact.py` - Evaluador Compacto
Evalúa modelos fine-tuneados en datos de test con métricas completas.

**Características:**
- Compatibilidad automática con datasets del procesador
- Generación automática de prompts si no existen
- Métricas por categoría (precision, recall, F1)
- Comparación de múltiples modelos
- Exportación CSV y JSON

## 🛠️ Instalación

### Prerequisitos
1. **Python 3.8+**
2. **CUDA GPU** (recomendado para entrenamiento)
3. **Ollama** instalado y ejecutándose

### Instalar Ollama
```bash
# Linux/Mac
curl -fsSL https://ollama.ai/install.sh | sh

# Verificar instalación
ollama --version
```

### Instalar dependencias Python
```bash
# Clonar repositorio
git clone <tu-repositorio>
cd cog-pipeline

# Instalar dependencias
pip install -r requirements.txt

# Para entornos con GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📊 Datos Requeridos

Coloca los siguientes archivos en `./cog_data/`:

```
cog_data/
├── cog-24.def.tab          # Definiciones COG
├── cog-24.org.csv          # Información de organismos
├── cog-24.cog.csv          # Mappings proteína-COG
└── COGorg24.faa.gz         # Secuencias de proteínas
```

**Descargar datos:**
```bash
# Crear directorio
mkdir cog_data && cd cog_data

# Descargar desde NCBI COG database
wget https://ftp.ncbi.nlm.nih.gov/pub/COG/COG2020/data/cog-24.def.tab
wget https://ftp.ncbi.nlm.nih.gov/pub/COG/COG2020/data/cog-24.org.csv  
wget https://ftp.ncbi.nlm.nih.gov/pub/COG/COG2020/data/cog-24.cog.csv
wget https://ftp.ncbi.nlm.nih.gov/pub/COG/COG2020/data/COGorg24.faa.gz
```

## 🔄 Flujo de Trabajo Completo

### Paso 1: Procesar Datos COG
```bash
# Generar datasets train/test balanceados
python cog_processor_fixed.py \
    --train-sequences 8000 \
    --test-sequences 2000 \
    --use-functional \
    --verbose

# Resultado: cog_train_functional.csv, cog_test_functional.csv
```

### Paso 2: Fine-tuning de Modelos
```bash
# Descargar modelo base
ollama pull llama3.2:3b

# Entrenamiento básico zero-shot
python cog_finetuner_compact.py \
    --model llama3.2:3b \
    --training-file cog_train_functional.csv \
    --output-model cog-expert \
    --max-steps 500 \
    --use-zero

# Entrenamiento completo (zero-shot + one-shot + LoRA + QLoRA)
python cog_finetuner_compact.py \
    --model llama3.2:3b \
    --training-file cog_train_functional.csv \
    --output-model cog-expert \
    --max-steps 800 \
    --use-dual \
    --dual-quantization

# Con optimización automática
python cog_finetuner_compact.py \
    --model llama3.2:3b \
    --training-file cog_train_functional.csv \
    --output-model cog-optimized \
    --max-steps 600 \
    --grid-search comprehensive
```

### Paso 3: Evaluación de Modelos
```bash
# Evaluar modelos entrenados
python cog_evaluator_compact.py \
    --models cog-expert-zero cog-expert-one \
    --test-file cog_test_functional.csv

# Evaluación rápida (50 muestras)
python cog_evaluator_compact.py \
    --models cog-expert \
    --test-file cog_test_functional.csv \
    --max-samples 50

# Ver modelos disponibles
python cog_evaluator_compact.py --list-models
```

## ⚙️ Configuración Avanzada

### Parámetros del Procesador
```bash
python cog_processor_fixed.py \
    --data-dir ./cog_data \
    --train-sequences 10000 \
    --test-sequences 2500 \
    --max-membership 1 \        # Solo alta calidad
    --min-bit-score 100 \       # Score mínimo alto
    --min-length 50 \           # Secuencias mínimas
    --max-length 1000 \         # Secuencias máximas
    --use-functional \          # Usar footprints
    --verbose
```

### Parámetros del Fine-tuner
```bash
python cog_finetuner_compact.py \
    --model llama3.2:3b \
    --training-file cog_train.csv \
    --output-model cog-custom \
    --max-steps 1000 \          # Más pasos
    --use-one \                 # One-shot prompts
    --use-lora \                # LoRA (16-bit)
    --grid-search quick         # Optimización rápida
```

### Parámetros del Evaluador
```bash
python cog_evaluator_compact.py \
    --models cog-model1 cog-model2 \
    --test-file test_data.csv \
    --max-samples 1000 \        # Limitar muestras
    --timeout 120 \             # Timeout por predicción
    --output-prefix evaluation  # Prefijo archivos
```

## 📈 Análisis de Resultados

### Archivos Generados

**Procesador:**
- `cog_train_functional.csv` - Datos de entrenamiento
- `cog_test_functional.csv` - Datos de test
- `protein_footprints.json` - Información de footprints

**Fine-tuner:**
- `training_qlora_YYYYMMDD_HHMMSS/` - Directorio de entrenamiento
- `training_info.json` - Metadatos del entrenamiento
- Modelos automáticamente importados a Ollama

**Evaluador:**
- `cog_evaluation_YYYYMMDD_HHMMSS.csv` - Resultados detallados
- `cog_evaluation_metrics_YYYYMMDD_HHMMSS.json` - Métricas completas

### Análisis en Python
```python
import pandas as pd
import json

# Cargar resultados de evaluación
df = pd.read_csv('cog_evaluation_20250802_143022.csv')
with open('cog_evaluation_metrics_20250802_143022.json', 'r') as f:
    metrics = json.load(f)

# Accuracy por modelo
for model, data in metrics['metrics']['individual_models'].items():
    print(f"{model}: {data['accuracy']:.3f}")

# Categorías más difíciles
errors = df[~df['correct']].groupby('true_category').size()
print("Categorías con más errores:")
print(errors.sort_values(ascending=False).head())

# Matriz de confusión
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(df['true_category'], df['predicted_category'])
sns.heatmap(cm, annot=True, fmt='d')
```

## 🐛 Troubleshooting

### Problemas Comunes

**1. Error "Modelo no encontrado"**
```bash
# Verificar modelos disponibles
ollama list

# Descargar modelo si no existe
ollama pull llama3.2:3b
```

**2. Error de memoria GPU**
```bash
# Usar QLoRA (4-bit) en lugar de LoRA
--use-qlora  # (por defecto)

# Reducir batch size en el código:
# per_device_train_batch_size: 1
```

**3. Datos COG no encontrados**
```bash
# Verificar estructura de directorios
ls -la cog_data/

# Re-descargar datos si es necesario
cd cog_data && wget <urls>
```

**4. Entrenamiento muy lento**
```bash
# Reducir pasos
--max-steps 200

# Usar menos muestras para pruebas
--max-samples 100
```

### Comandos de Diagnóstico
```bash
# Preview de datos
python cog_evaluator_compact.py --test-file data.csv --preview-data

# Verificar compatibilidad de modelos
python cog_finetuner_compact.py --list-models

# Test rápido (10 muestras)
python cog_evaluator_compact.py --models cog-expert --test-file test.csv --max-samples 10
```

## 📊 Métricas y Benchmarks

### Categorías COG (23 total)
```
J: Translation, ribosomal structure    A: RNA processing and modification
K: Transcription                       L: Replication, recombination, repair
B: Chromatin structure                 D: Cell cycle control, cell division
O: Molecular chaperones               M: Cell wall/membrane biogenesis
N: Cell motility                      P: Inorganic ion transport
T: Signal transduction                U: Intracellular trafficking
V: Defense mechanisms                 W: Extracellular structures
C: Energy production                  G: Carbohydrate metabolism
E: Amino acid metabolism              F: Nucleotide metabolism
H: Coenzyme metabolism                I: Lipid metabolism
Q: Secondary metabolites              R: General function prediction
S: Function unknown
```

### Performance Esperado
- **Accuracy**: 75-85% (con configuración optimizada)
- **F1-Score macro**: 70-80%
- **Tiempo de entrenamiento**: 1-3 horas (GPU)
- **Tiempo de predicción**: ~2s por proteína

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🙏 Agradecimientos

- **NCBI COG Database** - Datos de proteínas y clasificaciones
- **Ollama** - Framework de modelos de lenguaje local
- **Unsloth** - Framework optimizado para fine-tuning
- **Hugging Face** - Transformers y datasets

## 📚 Referencias

1. Tatusov, R.L., et al. (2000). The COG database: a tool for genome-scale analysis of protein functions and evolution.
2. Galperin, M.Y., et al. (2021). COG database update: focus on microbial diversity.
3. Lin, X., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure.

## 📞 Soporte

Para reportar bugs o solicitar features:
- **Issues**: [GitHub Issues](link-to-issues)
- **Email**: tu-email@example.com
- **Documentación**: [Wiki](link-to-wiki)

---

**Desarrollado con ❤️ para la comunidad de bioinformática**
