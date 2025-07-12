import os
import pandas as pd
import numpy as np
import requests
import json
import time
import re
import argparse
import sys
import pickle
from typing import Dict, List, Optional
import warnings
import logging
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OllamaClient:
    """Cliente para interactuar con Ollama API"""
    
    def __init__(self, host="localhost", port=11434):
        self.base_url = f"http://{host}:{port}"
        self.api_url = f"{self.base_url}/api"
        self.available_models = self._get_available_models()
        
        logger.info(f"🦙 Cliente Ollama inicializado: {self.base_url}")
        logger.info(f"📊 Modelos Ollama disponibles: {len(self.available_models)}")
    
    def _get_available_models(self) -> List[str]:
        """Obtiene lista de modelos disponibles en Ollama"""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except Exception as e:
            logger.warning(f"⚠️ Error conectando con Ollama: {e}")
            return []
    
    def is_available(self) -> bool:
        """Verifica si Ollama está disponible"""
        return len(self.available_models) > 0
    
    def is_model_available(self, model_name: str) -> bool:
        """Verifica si un modelo específico está disponible"""
        return model_name in self.available_models
    
    def generate_text(self, model: str, prompt: str, **kwargs) -> Optional[Dict]:
        """Genera texto usando Ollama"""
        if not self.is_model_available(model):
            logger.error(f"❌ Modelo {model} no disponible en Ollama")
            return None
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        try:
            start_time = time.time()
            timeout = getattr(self, 'timeout', 10000)  # Changed default timeout to 10000
            response = requests.post(f"{self.api_url}/generate", json=data, timeout=timeout)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                result['response_time'] = end_time - start_time
                return result
            else:
                logger.error(f"❌ Error Ollama {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error en generación Ollama: {e}")
            return None


class FunctionalCategoryPromptGenerator:
    """Generador de prompts para categorías funcionales COG"""
    
    def __init__(self):
        # Definir categorías COG para los prompts
        self.cog_categories = {
            'J': 'Translation, ribosomal structure and biogenesis',
            'A': 'RNA processing and modification',
            'K': 'Transcription', 
            'L': 'Replication, recombination and repair',
            'B': 'Chromatin structure and dynamics',
            'D': 'Cell cycle control, cell division',
            'O': 'Molecular chaperones and related functions',
            'M': 'Cell wall/membrane/envelope biogenesis',
            'N': 'Cell motility',
            'P': 'Inorganic ion transport and metabolism',
            'T': 'Signal transduction mechanisms',
            'U': 'Intracellular trafficking, secretion',
            'V': 'Defense mechanisms',
            'W': 'Extracellular structures',
            'C': 'Energy production and conversion',
            'G': 'Carbohydrate transport and metabolism',
            'E': 'Amino acid transport and metabolism',
            'F': 'Nucleotide transport and metabolism',
            'H': 'Coenzyme transport and metabolism',
            'I': 'Lipid transport and metabolism',
            'Q': 'Secondary metabolites biosynthesis',
            'R': 'General function prediction only',
            'S': 'Function unknown'
        }
    
    def generate_functional_category_prompt(self, sequence: str) -> str:
        """Genera prompt para predicción de categorías COG"""
        # CHANGED: No longer insert spaces every 3 amino acids, keep sequence as is
        
        prompt = f"""You are a protein function prediction expert. Analyze protein sequences and predict their COG functional category.

COG Categories:
J: Translation, ribosomal structure and biogenesis
A: RNA processing and modification  
K: Transcription
L: Replication, recombination and repair
B: Chromatin structure and dynamics
D: Cell cycle control, cell division
O: Molecular chaperones and related functions
M: Cell wall/membrane/envelope biogenesis
N: Cell motility
P: Inorganic ion transport and metabolism
T: Signal transduction mechanisms
U: Intracellular trafficking, secretion
V: Defense mechanisms
W: Extracellular structures
C: Energy production and conversion
G: Carbohydrate transport and metabolism
E: Amino acid transport and metabolism
F: Nucleotide transport and metabolism
H: Coenzyme transport and metabolism
I: Lipid transport and metabolism
Q: Secondary metabolites biosynthesis
R: General function prediction only
S: Function unknown

Now analyze this protein:
Protein sequence: {sequence}
Analysis: [Provide brief analysis of sequence characteristics]
Prediction: [Single letter only]"""
        
        return prompt
    
    def extract_prediction_from_response(self, response: str) -> str:
        """Extrae la predicción de la respuesta del LLM"""
        response = response.strip()
        
        # Buscar patrón "Prediction: [LETRA]"
        prediction_match = re.search(r'Prediction:\s*([A-S])\b', response, re.IGNORECASE)
        if prediction_match:
            return prediction_match.group(1).upper()
        
        # Buscar solo la letra al final
        letter_match = re.search(r'\b([A-S])\s*$', response)
        if letter_match:
            return letter_match.group(1).upper()
        
        # Buscar patrones alternativos
        alternative_patterns = [
            r'category\s+([A-S])\b',
            r'class\s+([A-S])\b',
            r'answer\s+([A-S])\b',
            r'^([A-S])\s*$'
        ]
        
        for pattern in alternative_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()
        
        return 'S'  # Function unknown por defecto


class OllamaFunctionalPredictor:
    """Predictor de categorías funcionales usando solo modelos de Ollama"""
    
    def __init__(self, ollama_host="localhost", ollama_port=11434, timeout=10000):  # Changed default timeout to 10000
        # Configuración óptima para clasificación de proteínas
        self.optimal_params = {
            'temperature': 0.08,
            'top_p': 0.85,
            'top_k': 30
        }
        
        self.timeout = timeout
        
        # Cliente Ollama
        self.ollama_client = OllamaClient(ollama_host, ollama_port)
        self.ollama_client.timeout = timeout
        
        # Generador de prompts
        self.prompt_generator = FunctionalCategoryPromptGenerator()
        
        # Categorías estándar COG
        self.standard_categories = ['J', 'A', 'K', 'L', 'B', 'D', 'O', 'M', 'N', 'P', 
                                  'T', 'U', 'V', 'W', 'C', 'G', 'E', 'F', 'H', 'I', 
                                  'Q', 'R', 'S']
        
        # Información de modelos
        self.model_info = {}
        
        logger.info(f"🦙 Predictor funcional Ollama inicializado")
        logger.info(f"🎯 Parámetros óptimos: {self.optimal_params}")
        logger.info(f"⏱️ Timeout: {timeout}s")
    
    def discover_available_models(self):
        """Descubre modelos de Ollama disponibles"""
        if not self.ollama_client.is_available():
            raise RuntimeError("❌ No hay modelos de Ollama disponibles")
        
        logger.info(f"🔍 Modelos Ollama encontrados:")
        for model_name in self.ollama_client.available_models:
            self.model_info[model_name] = {
                'type': self.classify_model_type(model_name),
                'backend': 'ollama'
            }
            logger.info(f"   🦙 {model_name}")
        
        logger.info(f"📊 Total: {len(self.model_info)} modelos")
        return self.model_info
    
    def classify_model_type(self, model_name):
        """Clasifica el tipo de modelo basándose en el nombre"""
        model_lower = model_name.lower()
        
        if 'embed' in model_lower:
            return 'embedding'
        elif 'deepseek' in model_lower:
            return 'deepseek'
        elif 'llama' in model_lower:
            return 'llama'
        elif 'mistral' in model_lower:
            return 'mistral'
        elif 'qwen' in model_lower:
            return 'qwen'
        elif 'gemma' in model_lower:
            return 'gemma'
        elif 'phi' in model_lower:
            return 'phi'
        else:
            return 'llm'
    
    def get_generation_params(self, model_name):
        """Obtiene parámetros optimizados por modelo"""
        model_lower = model_name.lower()
        
        # Parámetros específicos por familia
        if 'mistral' in model_lower:
            return {'temperature': 0.05, 'top_p': 0.85, 'top_k': 30}
        elif 'deepseek' in model_lower:
            return {'temperature': 0.08, 'top_p': 0.82, 'top_k': 28}
        elif 'llama' in model_lower:
            if '1b' in model_lower or '3b' in model_lower:
                return {'temperature': 0.10, 'top_p': 0.87, 'top_k': 32}
            else:
                return {'temperature': 0.06, 'top_p': 0.83, 'top_k': 25}
        elif 'qwen' in model_lower:
            return {'temperature': 0.07, 'top_p': 0.82, 'top_k': 28}
        elif 'gemma' in model_lower:
            return {'temperature': 0.10, 'top_p': 0.85, 'top_k': 32}
        elif 'phi' in model_lower:
            return {'temperature': 0.12, 'top_p': 0.87, 'top_k': 35}
        else:
            return self.optimal_params
    
    def load_csv_data(self, csv_path):
        """Carga datos desde el archivo CSV generado por el extractor COG"""
        logger.info(f"📊 Cargando datos desde {csv_path}...")
        
        try:
            df = pd.read_csv(csv_path)
            
            # Verificar columnas esperadas
            if 'sequence' not in df.columns or 'category' not in df.columns:
                raise ValueError("❌ El CSV debe tener columnas 'sequence' y 'category'")
            
            sequences = df['sequence'].tolist()
            true_categories = df['category'].tolist()
            
            logger.info(f"✅ Dataset completo cargado:")
            logger.info(f"   📈 Total de secuencias: {len(sequences):,}")
            logger.info(f"   🏷️ Categorías únicas: {len(set(true_categories))} ({sorted(set(true_categories))})")
            
            # Mostrar distribución completa
            category_counts = pd.Series(true_categories).value_counts()
            logger.info(f"📊 Distribución completa por categoría:")
            for category, count in category_counts.items():
                percentage = (count / len(sequences)) * 100
                logger.info(f"   {category}: {count:,} secuencias ({percentage:.1f}%)")
            
            # Mostrar estadísticas de secuencias
            seq_lengths = [len(seq) for seq in sequences]
            avg_length = sum(seq_lengths) / len(seq_lengths)
            min_length = min(seq_lengths)
            max_length = max(seq_lengths)
            
            logger.info(f"📏 Estadísticas de longitud de secuencias:")
            logger.info(f"   Promedio: {avg_length:.1f} aminoácidos")
            logger.info(f"   Rango: {min_length} - {max_length} aminoácidos")
            
            logger.info(f"🎯 Procesando TODAS las {len(sequences):,} secuencias automáticamente")
            
            return sequences, true_categories
            
        except Exception as e:
            logger.error(f"❌ Error cargando CSV: {e}")
            raise
    
    def predict_with_model(self, model_name, sequences):
        """Hace predicciones con un modelo específico"""
        logger.info(f"🦙 Prediciendo con {model_name}...")
        
        predictions = []
        detailed_responses = []  # Para guardar inputs y outputs
        generation_params = self.get_generation_params(model_name)
        
        # Barra de progreso
        progress = tqdm(sequences, 
                       desc=f"🦙 {model_name[:25]}...",
                       unit="seq")
        
        successful = 0
        failed = 0
        total_time = 0
        
        for i, sequence in enumerate(progress):
            start_time = time.time()
            
            try:
                # Generar prompt
                prompt = self.prompt_generator.generate_functional_category_prompt(sequence)
                
                # Generar respuesta
                result = self.ollama_client.generate_text(model_name, prompt, **generation_params)
                
                if result and 'response' in result:
                    response = result['response']
                    prediction = self.prompt_generator.extract_prediction_from_response(response)
                    predictions.append(prediction)
                    successful += 1
                    
                    # Guardar detalles para CSV de inputs/outputs
                    detailed_responses.append({
                        'sequence_index': i,
                        'sequence': sequence,
                        'prompt': prompt,
                        'raw_response': response,
                        'extracted_prediction': prediction,
                        'response_time': result.get('response_time', 0),
                        'model_name': model_name,
                        'generation_params': str(generation_params)
                    })
                else:
                    predictions.append('S')  # Function unknown
                    failed += 1
                    
                    # Guardar también los fallos
                    detailed_responses.append({
                        'sequence_index': i,
                        'sequence': sequence,
                        'prompt': prompt,
                        'raw_response': 'ERROR: No response from model',
                        'extracted_prediction': 'S',
                        'response_time': 0,
                        'model_name': model_name,
                        'generation_params': str(generation_params)
                    })
                    
            except Exception as e:
                logger.warning(f"Error en secuencia {i}: {e}")
                predictions.append('S')
                failed += 1
                
                # Guardar error
                detailed_responses.append({
                    'sequence_index': i,
                    'sequence': sequence,
                    'prompt': 'ERROR: Could not generate prompt',
                    'raw_response': f'ERROR: {str(e)}',
                    'extracted_prediction': 'S',
                    'response_time': 0,
                    'model_name': model_name,
                    'generation_params': str(generation_params)
                })
            
            # Actualizar progreso
            elapsed = time.time() - start_time
            total_time += elapsed
            
            progress.set_postfix({
                "✅": successful,
                "❌": failed,
                "⏱️": f"{elapsed:.1f}s"
            })
        
        success_rate = (successful / len(sequences)) * 100
        avg_time = total_time / len(sequences)
        
        logger.info(f"✅ {model_name}: {successful}/{len(sequences)} exitosas "
                   f"({success_rate:.1f}%), {avg_time:.1f}s/seq")
        
        return predictions, detailed_responses
    
    def predict_all_models(self, sequences, max_models=None):
        """Hace predicciones con todos los modelos disponibles para TODAS las secuencias"""
        total_sequences = len(sequences)
        total_models = len(self.model_info)
        
        logger.info(f"🚀 INICIANDO PREDICCIONES AUTOMÁTICAS:")
        logger.info(f"   📊 Dataset: {total_sequences:,} secuencias")
        logger.info(f"   🦙 Modelos: {total_models} disponibles")
        
        all_predictions = {}
        all_detailed_responses = []  # Para el CSV de inputs/outputs
        models_to_use = list(self.model_info.keys())
        
        if max_models:
            models_to_use = models_to_use[:max_models]
            logger.info(f"🎯 Limitando evaluación a {max_models} modelos")
        else:
            logger.info(f"🎯 Evaluando TODOS los {len(models_to_use)} modelos disponibles")
        
        # Progreso general
        main_progress = tqdm(models_to_use, 
                           desc="🎯 Progreso general", 
                           unit="modelo")
        
        successful_models = 0
        failed_models = 0
        total_predictions_made = 0
        
        for model_idx, model_name in enumerate(main_progress, 1):
            main_progress.set_postfix({
                "Modelo": f"{model_idx}/{len(models_to_use)}",
                "Actual": model_name[:15],
                "✅": successful_models,
                "❌": failed_models
            })
            
            try:
                start_time = time.time()
                predictions, detailed_responses = self.predict_with_model(model_name, sequences)
                prediction_time = time.time() - start_time
                
                all_predictions[model_name] = {
                    'predictions': predictions,
                    'prediction_time': prediction_time,
                    'samples_per_second': len(sequences) / prediction_time,
                    'model_type': self.model_info[model_name]['type'],
                    'success_rate': predictions.count('S') / len(predictions) if predictions else 0
                }
                
                # Agregar respuestas detalladas para el CSV de inputs/outputs
                all_detailed_responses.extend(detailed_responses)
                
                successful_models += 1
                total_predictions_made += len(predictions)
                
                # Actualizar progreso con estadísticas
                main_progress.set_description(f"🎯 {successful_models}/{len(models_to_use)} completados")
                
            except Exception as e:
                logger.error(f"❌ Error con {model_name}: {e}")
                failed_models += 1
                continue
        
        main_progress.close()
        
        # Resumen final detallado
        logger.info(f"🎉 PREDICCIONES AUTOMÁTICAS COMPLETADAS:")
        logger.info(f"   ✅ Modelos exitosos: {successful_models}/{total_models}")
        logger.info(f"   ❌ Modelos fallidos: {failed_models}")
        logger.info(f"   📈 Total predicciones generadas: {total_predictions_made:,}")
        logger.info(f"   📊 Promedio por modelo: {total_predictions_made//successful_models if successful_models > 0 else 0:,} predicciones")
        logger.info(f"   🔍 Respuestas detalladas capturadas: {len(all_detailed_responses):,}")
        
        if successful_models > 0:
            # Mostrar velocidades promedio
            total_time = sum(result['prediction_time'] for result in all_predictions.values())
            avg_speed = total_predictions_made / total_time if total_time > 0 else 0
            logger.info(f"   ⚡ Velocidad promedio: {avg_speed:.1f} predicciones/segundo")
        
        return all_predictions, all_detailed_responses
    
    def save_results(self, all_predictions, sequences, true_categories, output_path, detailed_responses=None):
        """Guarda resultados en formato CSV y pickle"""
        logger.info(f"💾 Guardando resultados completos en {output_path}...")
        
        # Crear DataFrame base
        results_data = {
            'sequence': sequences,
            'true_category': true_categories
        }
        
        # Agregar predicciones de cada modelo
        model_count = 0
        for model_name, model_results in all_predictions.items():
            if isinstance(model_results, dict) and 'predictions' in model_results:
                results_data[f'pred_{model_name}'] = model_results['predictions']
                model_count += 1
        
        # Crear DataFrame y guardar CSV
        df_results = pd.DataFrame(results_data)
        df_results.to_csv(output_path, index=False)
        
        logger.info(f"✅ Resultados CSV guardados:")
        logger.info(f"   📊 {df_results.shape[0]:,} filas × {df_results.shape[1]} columnas")
        logger.info(f"   🦙 {model_count} modelos evaluados")
        logger.info(f"   📈 {df_results.shape[0] * model_count:,} predicciones totales")
        
        # ADDED: Save results in pickle format for compatibility with benchmarking code
        pickle_path = output_path.replace('.csv', '.pkl')
        logger.info(f"💾 Guardando resultados en formato pickle en {pickle_path}...")
        
        # Create pickle-compatible structure
        pickle_data = {
            'all_predictions': all_predictions,
            'sequences': sequences,
            'true_categories': true_categories,
            'model_info': self.model_info,
            'standard_categories': self.standard_categories
        }
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(pickle_data, f)
        
        logger.info(f"✅ Resultados pickle guardados:")
        logger.info(f"   📄 Archivo: {pickle_path}")
        logger.info(f"   🔧 Compatible con códigos de benchmarking existentes")
        
        # Guardar CSV de inputs/outputs detallados
        if detailed_responses:
            detailed_output_path = output_path.replace('.csv', '_inputs_outputs.csv')
            logger.info(f"💾 Guardando inputs/outputs detallados en {detailed_output_path}...")
            
            df_detailed = pd.DataFrame(detailed_responses)
            df_detailed.to_csv(detailed_output_path, index=False)
            
            logger.info(f"✅ Inputs/Outputs detallados guardados:")
            logger.info(f"   📝 {len(detailed_responses):,} interacciones capturadas")
            logger.info(f"   🔍 Columnas: sequence, prompt, raw_response, extracted_prediction, model_name, etc.")
            logger.info(f"   📄 Archivo: {detailed_output_path}")
        
        # Crear resumen de rendimiento expandido
        summary_path = output_path.replace('.csv', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Resumen de Predicciones COG - Procesamiento Automático Completo\n")
            f.write("=" * 65 + "\n\n")
            f.write(f"Dataset procesado automáticamente:\n")
            f.write(f"  Total de secuencias: {len(sequences):,}\n")
            f.write(f"  Modelos evaluados: {model_count}\n")
            f.write(f"  Predicciones generadas: {len(sequences) * model_count:,}\n")
            f.write(f"  Categorías únicas: {len(set(true_categories))}\n")
            if detailed_responses:
                f.write(f"  Interacciones detalladas capturadas: {len(detailed_responses):,}\n")
            f.write("\n")
            
            # Distribución de categorías reales
            category_counts = pd.Series(true_categories).value_counts()
            f.write("Distribución del dataset:\n")
            f.write("-" * 25 + "\n")
            for category, count in category_counts.items():
                percentage = (count / len(sequences)) * 100
                f.write(f"  {category}: {count:,} secuencias ({percentage:.1f}%)\n")
            f.write("\n")
            
            f.write("Rendimiento por modelo:\n")
            f.write("-" * 25 + "\n")
            for model_name, model_results in all_predictions.items():
                if isinstance(model_results, dict):
                    time_taken = model_results.get('prediction_time', 0)
                    sps = model_results.get('samples_per_second', 0)
                    model_type = model_results.get('model_type', 'unknown')
                    success_rate = model_results.get('success_rate', 0)
                    f.write(f"{model_name} ({model_type}):\n")
                    f.write(f"  Tiempo total: {time_taken:.1f}s\n")
                    f.write(f"  Velocidad: {sps:.1f} seq/s\n")
                    f.write(f"  Secuencias procesadas: {len(sequences):,}\n")
                    f.write(f"  Tasa de respuesta 'S': {success_rate:.1%}\n\n")
            
            # Información adicional
            f.write("Archivos generados:\n")
            f.write("-" * 18 + "\n")
            f.write(f"  1. {output_path} - Predicciones finales (CSV)\n")
            f.write(f"  2. {pickle_path} - Predicciones finales (Pickle)\n")
            if detailed_responses:
                f.write(f"  3. {detailed_output_path} - Inputs/Outputs detallados\n")
            f.write(f"  4. {summary_path} - Este resumen\n\n")
            
            f.write("Configuración utilizada:\n")
            f.write("-" * 22 + "\n")
            f.write(f"  Procesamiento: Automático (dataset completo)\n")
            f.write(f"  Modo: Sin limitaciones de muestras\n")
            f.write(f"  Parámetros: Optimizados por familia de modelo\n")
            f.write(f"  Timeout: {self.timeout}s\n")
            f.write(f"  Captura detallada: {'Habilitada' if detailed_responses else 'Deshabilitada'}\n")
            f.write(f"  Formato secuencia: Sin espacios (continuo)\n")
        
        logger.info(f"📊 Resumen detallado guardado: {summary_path}")
        
        # Mostrar información sobre los archivos generados
        logger.info(f"\n📁 ARCHIVOS GENERADOS:")
        logger.info(f"   1. 📊 {output_path} - Predicciones finales (CSV)")
        logger.info(f"   2. 🔧 {pickle_path} - Predicciones finales (Pickle)")
        if detailed_responses:
            logger.info(f"   3. 🔍 {detailed_output_path} - Inputs/Outputs detallados")
        logger.info(f"   4. 📋 {summary_path} - Resumen de rendimiento")
        
        if detailed_responses:
            logger.info(f"\n🔍 PARA REVISAR RESPUESTAS DE MODELOS:")
            logger.info(f"   📖 Abrir: {detailed_output_path}")
            logger.info(f"   📝 Contiene: prompts completos + respuestas completas + predicciones extraídas")
            logger.info(f"   🔬 Útil para: debugging, análisis de prompts, mejora de extracción")
        
        logger.info(f"\n🔧 COMPATIBILIDAD PICKLE:")
        logger.info(f"   📄 Archivo: {pickle_path}")
        logger.info(f"   🎯 Compatible con códigos de benchmarking existentes")
        logger.info(f"   📊 Estructura: all_predictions, sequences, true_categories, model_info")


def parse_arguments():
    """Parsea argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='Predictor de Categorías Funcionales COG usando Ollama con captura de inputs/outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Uso básico (procesa TODAS las secuencias automáticamente + captura inputs/outputs)
  python ollama_functional_predictor.py --csv cog_categories_balanced_460.csv
  
  # Con dataset completo
  python ollama_functional_predictor.py --csv cog_categories_complete.csv
  
  # Solo algunos modelos
  python ollama_functional_predictor.py --csv cog_categories_balanced_460.csv --models llama3.2:3b mistral:7b
  
  # Salida personalizada
  python ollama_functional_predictor.py --csv cog_categories_balanced_460.csv --output results/predictions.csv
  
  # Limitar número de modelos evaluados
  python ollama_functional_predictor.py --csv cog_categories_balanced_460.csv --max-models 3

Archivos generados automáticamente:
  1. [output].csv - Predicciones finales (sequence, true_category, pred_model1, pred_model2, ...)
  2. [output].pkl - Predicciones en formato pickle (compatible con benchmarking)
  3. [output]_inputs_outputs.csv - Inputs/outputs detallados (prompts completos + respuestas)
  4. [output]_summary.txt - Resumen de rendimiento

Cambios en esta versión:
  - Secuencias sin espacios (formato continuo: MKKIAVFVP...)
  - Timeout aumentado a 10000 segundos
  - Guardado automático en formato pickle para compatibilidad
        """
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Archivo CSV con secuencias (columnas: sequence, category)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='cog_predictions.csv',
        help='Archivo de salida para las predicciones'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help='Modelos específicos a usar (ej: --models llama3.2:3b mistral:7b)'
    )
    
    parser.add_argument(
        '--max-models',
        type=int,
        default=None,
        help='Número máximo de modelos a evaluar (opcional)'
    )
    
    parser.add_argument(
        '--ollama-host',
        type=str,
        default='localhost',
        help='Host de Ollama'
    )
    
    parser.add_argument(
        '--ollama-port',
        type=int,
        default=11434,
        help='Puerto de Ollama'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=10000,
        help='Timeout en segundos por predicción (default: 10000)'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='Solo listar modelos disponibles'
    )
    
    return parser.parse_args()


def main():
    """Función principal"""
    args = parse_arguments()
    
    # Inicializar predictor
    try:
        predictor = OllamaFunctionalPredictor(
            ollama_host=args.ollama_host,
            ollama_port=args.ollama_port,
            timeout=args.timeout
        )
        
        # Descubrir modelos
        predictor.discover_available_models()
        
        # Si solo se quiere listar modelos
        if args.list_models:
            logger.info("📋 Modelos de Ollama disponibles:")
            for model_name, info in predictor.model_info.items():
                logger.info(f"   🦙 {model_name} ({info['type']})")
            return
        
        # Filtrar modelos si se especificaron
        if args.models:
            filtered_models = {}
            for model in args.models:
                if model in predictor.model_info:
                    filtered_models[model] = predictor.model_info[model]
                else:
                    logger.warning(f"⚠️ Modelo {model} no encontrado")
            predictor.model_info = filtered_models
            logger.info(f"🎯 Usando {len(filtered_models)} modelos especificados")
        
    except RuntimeError as e:
        logger.error(f"❌ Error inicializando Ollama: {e}")
        sys.exit(1)
    
    # Cargar datos del CSV (automáticamente todas las secuencias)
    try:
        sequences, true_categories = predictor.load_csv_data(args.csv)
        
        logger.info(f"🎯 CONFIGURACIÓN AUTOMÁTICA:")
        logger.info(f"   📊 Secuencias a procesar: {len(sequences):,} (TODAS)")
        logger.info(f"   🦙 Modelos a evaluar: {len(predictor.model_info)}")
        logger.info(f"   📈 Total predicciones: {len(sequences) * len(predictor.model_info):,}")
        logger.info(f"   ⏱️ Timeout por predicción: {args.timeout}s")
        logger.info(f"   📝 Formato secuencia: Continuo (sin espacios)")
        
        # Estimación de tiempo
        estimated_time_minutes = (len(sequences) * len(predictor.model_info) * 3) / 60  # ~3 segundos por predicción
        if estimated_time_minutes > 60:
            time_str = f"{estimated_time_minutes/60:.1f} horas"
        else:
            time_str = f"{estimated_time_minutes:.1f} minutos"
        
        logger.info(f"   ⏱️ Tiempo estimado: ~{time_str}")
        
    except Exception as e:
        logger.error(f"❌ Error cargando datos: {e}")
        sys.exit(1)
    
    # Hacer predicciones con TODAS las secuencias
    try:
        logger.info("🚀 Iniciando predicciones con dataset completo...")
        
        # FIX: Unpack the tuple returned by predict_all_models
        all_predictions, all_detailed_responses = predictor.predict_all_models(
            sequences, 
            max_models=args.max_models
        )
        
        if not all_predictions:
            logger.error("❌ No se generaron predicciones")
            sys.exit(1)
        
        # Guardar resultados - FIX: Pass the unpacked values correctly
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        predictor.save_results(all_predictions, sequences, true_categories, args.output, all_detailed_responses)
        
        logger.info("🎉 Proceso completado exitosamente!")
        logger.info(f"📁 Resultados en: {args.output}")
        logger.info(f"🔧 Pickle compatible en: {args.output.replace('.csv', '.pkl')}")
        logger.info(f"📊 Procesadas {len(sequences):,} secuencias con {len(all_predictions)} modelos")
        
        # Mostrar próximos pasos
        print(f"\n💡 Ejemplos de uso automático:")
        print(f"   # Procesar TODAS las secuencias (automático)")
        print(f"   python {__file__} --csv cog_categories_balanced_460.csv")
        print(f"")
        print(f"   # Con dataset completo")
        print(f"   python {__file__} --csv cog_categories_complete.csv")
        print(f"")
        print(f"   # Solo modelos específicos")
        print(f"   python {__file__} --csv cog_categories_balanced_460.csv --models llama3.2:3b mistral:7b")
        print(f"")
        print(f"   # Limitar evaluación a 3 modelos más rápidos")
        print(f"   python {__file__} --csv cog_categories_balanced_460.csv --max-models 3")
        print(f"")
        print(f"   # Salida personalizada")
        print(f"   python {__file__} --csv cog_categories_balanced_460.csv --output results/predictions.csv")
        print(f"")
        print(f"   # Timeout personalizado")
        print(f"   python {__file__} --csv cog_categories_balanced_460.csv --timeout 5000")
        print(f"")
        print(f"🎯 VENTAJAS DEL PROCESAMIENTO AUTOMÁTICO:")
        print(f"   ✅ Detecta automáticamente todas las secuencias del CSV")
        print(f"   ✅ No requiere especificar --max-samples")
        print(f"   ✅ Procesa el dataset completo sin limitaciones")
        print(f"   ✅ Muestra estadísticas detalladas automáticamente")
        print(f"   ✅ Estimación de tiempo precisa")
        print(f"   ✅ Progreso en tiempo real para cada modelo")
        print(f"   ✅ Guardado automático en CSV y Pickle")
        print(f"   ✅ Secuencias en formato continuo (sin espacios)")
        print(f"   ✅ Timeout aumentado a 10000s por defecto")
        print(f"")
        print(f"📊 RESULTADO:")
        print(f"   {len(sequences):,} secuencias × {len(all_predictions)} modelos = {len(sequences) * len(all_predictions):,} predicciones")
        print(f"")
        print(f"🔧 ARCHIVOS GENERADOS:")
        print(f"   📊 {args.output} - Predicciones CSV")
        print(f"   🔧 {args.output.replace('.csv', '.pkl')} - Predicciones Pickle (compatible con benchmarking)")
        print(f"   🔍 {args.output.replace('.csv', '_inputs_outputs.csv')} - Inputs/Outputs detallados")
        print(f"   📋 {args.output.replace('.csv', '_summary.txt')} - Resumen de rendimiento")
        
    except Exception as e:
        logger.error(f"❌ Error durante predicciones: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("🦙 Predictor de Categorías Funcionales COG con Ollama (Modificado)")
    print("=" * 65)
    print("🎯 PROCESAMIENTO AUTOMÁTICO:")
    print("   ✅ Detecta automáticamente todas las secuencias del CSV")
    print("   ✅ No requiere especificar límites de muestras")
    print("   ✅ Evalúa todos los modelos de Ollama disponibles")
    print("   ✅ Parámetros optimizados automáticamente por modelo")
    print("   ✅ Captura inputs/outputs detallados para revisión")
    print("   ✅ Secuencias en formato continuo (sin espacios)")
    print("   ✅ Timeout aumentado a 10000 segundos")
    print("   ✅ Guardado automático en CSV y Pickle")
    print("=" * 65)
    print()
    
    main()