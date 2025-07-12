# COG Protein Function Prediction Pipeline

## Descripción del Proyecto
Pipeline completo para predicción de categorías funcionales COG (Clusters of Orthologous Groups) en secuencias de proteínas utilizando modelos de lenguaje grandes (LLMs) a través de Ollama. El proyecto incluye herramientas para extracción de datos, preparación de datasets y evaluación comparativa de diferentes enfoques de prompting.

## Componentes del Pipeline

### 1. COG Extractor (`cog_extractor.py`)
**Propósito**: Extracción y preparación de datasets COG balanceados o completos  
**Entrada**: Archivos de base de datos COG (def, csv, fasta)  
**Salida**: Datasets CSV optimizados para ML (sequence,category)

### 2. Ollama Predictor Estándar (`ollama_predictor_one-shot.py`)
**Propósito**: Predicción de categorías COG con prompt que incluye ejemplo específico  
**Entrada**: Dataset CSV de secuencias  
**Salida**: Predicciones de múltiples modelos + análisis detallado

### 3. Ollama Predictor Sin Ejemplo (`ollama_predictor_zero-shot.py`)
**Propósito**: Predicción de categorías COG con prompt limpio (sin ejemplo)  
**Entrada**: Dataset CSV de secuencias  
**Salida**: Predicciones sin sesgo + análisis comparativo

### Modelos Instalados en Ollama
NAME                   ID              SIZE     
qwen3:latest           500a1f067a9f    5.2 GB   
deepseek-llm:latest    9aab369a853b    4.0 GB   
deepseek-r1:7b         755ced02ce7b    4.7 GB  
deepseek-r1:1.5b       e0979632db5a    1.1 GB   
qwen2.5:7b             845dbda0ea48    4.7 GB   
deepseek-llm:7b        9aab369a853b    4.0 GB  
llama3.2:3b            a80c4f17acd5    2.0 GB    
llama3.2:1b            baf6a787fdff    1.3 GB    
mistral:latest         3944fe81ec14    4.1 GB    

### Instalación
```bash
# 1. Clonar/descargar archivos del proyecto
git clone [repositorio] # o descargar archivos (Aun no subidos a git)

# 2. Instalar dependencias para cada componente
pip install -r requirements.txt

# O si prefieres usar conda
conda install --file requirements.txt

# 3. Instalar y configurar Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
ollama pull qwen3:latest           # 5.2 GB
ollama pull deepseek-llm:latest    # 4.0 GB  
ollama pull deepseek-r1:7b         # 4.7 GB
ollama pull deepseek-r1:1.5b       # 1.1 GB
ollama pull qwen2.5:7b             # 4.7 GB
ollama pull deepseek-llm:7b        # 4.0 GB
ollama pull llama3.2:3b            # 2.0 GB
ollama pull llama3.2:1b            # 1.3 GB
ollama pull mistral:latest         # 4.1 GB

# 4. Descargar archivos COG (/home/lab/Desktop/uwo/COG_LLMS/COG2024)
# - cog-24.def.tab
# - cog-24.cog.csv
# - COGorg24.faa.gz
```

## 📋 Uso Paso a Paso

### Paso 1: Preparar Dataset
```bash
# Crear dataset balanceado pequeño (para pruebas)
python cog_extractor.py /home/lab/Desktop/uwo/COG_LLMS/COG2024 --samples 460

# Crear dataset completo (hace un csv con todas las secuencias)
python cog_extractor.py /home/lab/Desktop/uwo/COG_LLMS/COG2024
```

**Salida**: `cog_categories_balanced_460.csv`, `cog_categories_balanced_*num-sample*.csv`, etc.


### Paso 2: Ejecutar Predicciones

#### Enfoque Estándar (Con Ejemplo)
```bash
# Evaluación con prompt que incluye ejemplo específico
python ollama_predictor_one-shot.py --csv cog_categories_balanced_460.csv --output results_one.csv
```

#### Enfoque Experimental (Sin Ejemplo)
```bash
# Evaluación con prompt limpio sin ejemplo
python ollama_predictor_zero-shot.py --csv cog_categories_balanced_460.csv --output results_zero.csv
```

### Paso 3: Análisis Comparativo
python benchmarking_ollama_fixed.py --predictions cog_predictions.pkl
```

**Salida**: Directorio Benchmark results con archivos que muestran metricas

## 📊 Estructura de Archivos del Proyecto

```
proyecto_cog/
├── cog_extractor.py                                                    # Extractor de datos COG
├── ollama_predictor_one-shot.py                                        # Predictor con ejemplo
├── ollama_predictor_zero-shot.py                                       # Predictor sin ejemplo
├── benchmarking_ollama_fixed.py                                        # Codigo de Evaluacion
├── requirements.txt                                                    # Dependencias del extractor
├── README.txt                                                          # Documentación
├── benchmark_results                                                   # Graficos y archivos con metricas de los modelos
└── /home/lab/Desktop/uwo/COG_LLMS/COG2024                              # Directorio datos COG
```

## Ejemplos de configuracion

### Optimización por Tamaño de Dataset
```bash
# Dataset pequeño (desarrollo rápido)
python cog_extractor.py /data/cog --samples 230  # ~10 por categoría

# Dataset mediano (evaluación estándar)
python cog_extractor.py /data/cog --samples 460  # ~20 por categoría

# Dataset grande (investigación)
python cog_extractor.py /data/cog --samples 2300  # ~100 por categoría
```

### Configuración de Modelos
```bash
# Evaluación rápida (modelos pequeños)
python ollama_predictor_*.py --csv data.csv --models llama3.2:3b qwen2.5:3b --max-models 2

# Evaluación completa (todos los modelos)
python ollama_predictor_*.py --csv data.csv

# Evaluación de alta calidad (modelos grandes)
python ollama_predictor_*.py --csv data.csv --models llama3.1:70b mistral:large
```

### Configuración de Timeout (Para que el modelo no se desconecte despues de pasar el tiempo definido)
```bash
# Para modelos rápidos
python ollama_predictor_*.py --csv data.csv --timeout 3000

# Para modelos lentos o complejos
python ollama_predictor_*.py --csv data.csv --timeout 15000
```

## Solución de Problemas Comunes

### Error: Archivos COG no encontrados
```bash
# Verificar estructura de directorio
ls /home/lab/Desktop/uwo/COG_LLMS/COG2024
# Debe contener: cog-24.def.tab, cog-24.cog.csv, COGorg24.faa.gz
```

### Error: Ollama no responde
```bash
# Reiniciar Ollama
pkill ollama
ollama serve

# Verificar modelos
ollama list
```