# Aplications-of-LLM-on-Functional-Annotation
Trabajo de titulo

cog2024 analyzer es el archivo con el cual se hace un vistazo a la base de datos de cog, sus distribuciones de datos y etc

-----------------------------------------------------------------------------------------------------------------------------

1. Data preparation console
2. hybrid loader 
3. benchmark multitasking

-----------------------------------------------------------------------------------------------------------------------------

Observaciones:
- Cambiar la temperatura y top_p, hasta ahora llama con la configuracion 'temperature': 0.2, 'top_p': 0.9 es la que posee mejor rendimiento en functional annotations por otro lado qwen tiene mejor rendimiento en pathways con una configuracion de 'temperature': 0.1, 'top_p': 0.85

    if '1b' in model_lower:
        auto_params = {'temperature': 0.2, 'top_p': 0.9}  # Más exploratorio
    else:
        auto_params = {'temperature': 0.15, 'top_p': 0.85}  # Más determinístico

revisar esa configuracion!!

1. Only prompt Usage
   1. Data_preparation
   2. Hybrid_loader
   3. Benchmark multitask

2. Rag Usage

3. Finetuning models
   1. Data_processor
   2. Finetuner
   3. Evaluation
