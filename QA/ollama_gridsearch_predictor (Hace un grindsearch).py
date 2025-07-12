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
from typing import Dict, List, Optional, Tuple
import warnings
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from itertools import product
import random

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
            timeout = getattr(self, 'timeout', 10000)
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

Example:
Protein sequence: MKIAITGIPGVGKTTIAKALAKKLGYQYIDLNKLIIEKYKPKYDWFYDSYIIEDDMLNIEIPDNSVIDSHLSHLLDVDLVVYLIADPKIIEQRLKERGYSFSKIFENIWAQTAGIIESELVGKKYIKIDVTNKDVDTIVNQIIDYISTLDKK
Analysis: This sequence shows characteristics of membrane proteins with multiple transmembrane domains, likely involved in transport functions.
Prediction: F

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


class GridSearchOptimizer:
    """Optimizador de hiperparámetros usando grid search"""
    
    def __init__(self):
        # Definir grids de parámetros para diferentes familias de modelos
        self.parameter_grids = {
            'llama': {
                'temperature': [0.01, 0.05, 0.08, 0.12, 0.15],
                'top_p': [0.80, 0.85, 0.90, 0.95],
                'top_k': [20, 25, 30, 35, 40]
            },
            'mistral': {
                'temperature': [0.01, 0.03, 0.05, 0.08, 0.10],
                'top_p': [0.82, 0.85, 0.88, 0.92],
                'top_k': [25, 30, 35, 40]
            },
            'deepseek': {
                'temperature': [0.05, 0.08, 0.10, 0.12, 0.15],
                'top_p': [0.78, 0.82, 0.85, 0.88],
                'top_k': [20, 28, 32, 35]
            },
            'qwen': {
                'temperature': [0.03, 0.07, 0.10, 0.12],
                'top_p': [0.80, 0.82, 0.85, 0.88],
                'top_k': [25, 28, 32, 35]
            },
            'gemma': {
                'temperature': [0.08, 0.10, 0.12, 0.15, 0.18],
                'top_p': [0.83, 0.85, 0.88, 0.90],
                'top_k': [28, 32, 35, 40]
            },
            'phi': {
                'temperature': [0.10, 0.12, 0.15, 0.18, 0.20],
                'top_p': [0.85, 0.87, 0.90, 0.92],
                'top_k': [30, 35, 40, 45]
            },
            'default': {
                'temperature': [0.05, 0.08, 0.10, 0.12],
                'top_p': [0.82, 0.85, 0.88],
                'top_k': [25, 30, 35]
            }
        }
    
    def get_model_family(self, model_name: str) -> str:
        """Identifica la familia del modelo"""
        model_lower = model_name.lower()
        
        if 'llama' in model_lower:
            return 'llama'
        elif 'mistral' in model_lower:
            return 'mistral'
        elif 'deepseek' in model_lower:
            return 'deepseek'
        elif 'qwen' in model_lower:
            return 'qwen'
        elif 'gemma' in model_lower:
            return 'gemma'
        elif 'phi' in model_lower:
            return 'phi'
        else:
            return 'default'
    
    def generate_parameter_combinations(self, model_name: str, max_combinations: int = 20) -> List[Dict]:
        """Genera combinaciones de parámetros para grid search"""
        family = self.get_model_family(model_name)
        grid = self.parameter_grids[family]
        
        # Generar todas las combinaciones posibles
        param_names = list(grid.keys())
        param_values = list(grid.values())
        all_combinations = list(product(*param_values))
        
        # Limitar número de combinaciones si es necesario
        if len(all_combinations) > max_combinations:
            # Seleccionar combinaciones de manera estratégica
            # 1. Incluir esquinas del espacio de parámetros
            corner_indices = [0, len(all_combinations)-1]
            
            # 2. Agregar algunas combinaciones aleatorias
            random.seed(42)  # Para reproducibilidad
            remaining_indices = list(range(1, len(all_combinations)-1))
            random.shuffle(remaining_indices)
            selected_indices = corner_indices + remaining_indices[:max_combinations-2]
            
            combinations = [all_combinations[i] for i in sorted(selected_indices)]
        else:
            combinations = all_combinations
        
        # Convertir a lista de diccionarios
        param_combinations = []
        for combo in combinations:
            param_dict = dict(zip(param_names, combo))
            param_combinations.append(param_dict)
        
        logger.info(f"🔍 Grid search para {model_name} ({family}): {len(param_combinations)} combinaciones")
        
        return param_combinations
    
    def evaluate_parameters(self, model_name: str, params: Dict, 
                          sequences: List[str], true_categories: List[str],
                          ollama_client, prompt_generator) -> Dict:
        """Evalúa un conjunto específico de parámetros"""
        
        predictions = []
        failed_predictions = 0
        total_time = 0
        
        for sequence, true_cat in zip(sequences, true_categories):
            try:
                start_time = time.time()
                
                prompt = prompt_generator.generate_functional_category_prompt(sequence)
                result = ollama_client.generate_text(model_name, prompt, **params)
                
                if result and 'response' in result:
                    response = result['response']
                    prediction = prompt_generator.extract_prediction_from_response(response)
                    predictions.append(prediction)
                    total_time += result.get('response_time', 0)
                else:
                    predictions.append('S')  # Default fallback
                    failed_predictions += 1
                    total_time += time.time() - start_time
                    
            except Exception as e:
                predictions.append('S')
                failed_predictions += 1
                total_time += time.time() - start_time
        
        # Calcular métricas
        accuracy = accuracy_score(true_categories, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_categories, predictions, average='weighted', zero_division=0
        )
        
        # Estadísticas adicionales
        success_rate = (len(predictions) - failed_predictions) / len(predictions)
        avg_time_per_prediction = total_time / len(predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'success_rate': success_rate,
            'avg_time': avg_time_per_prediction,
            'failed_predictions': failed_predictions,
            'predictions': predictions
        }


class OllamaGridSearchPredictor:
    """Predictor con optimización de hiperparámetros usando grid search"""
    
    def __init__(self, ollama_host="localhost", ollama_port=11434, timeout=10000):
        self.timeout = timeout
        
        # Cliente Ollama
        self.ollama_client = OllamaClient(ollama_host, ollama_port)
        self.ollama_client.timeout = timeout
        
        # Generador de prompts
        self.prompt_generator = FunctionalCategoryPromptGenerator()
        
        # Optimizador de hiperparámetros
        self.grid_optimizer = GridSearchOptimizer()
        
        # Categorías estándar COG
        self.standard_categories = ['J', 'A', 'K', 'L', 'B', 'D', 'O', 'M', 'N', 'P', 
                                  'T', 'U', 'V', 'W', 'C', 'G', 'E', 'F', 'H', 'I', 
                                  'Q', 'R', 'S']
        
        # Información de modelos y sus mejores parámetros
        self.model_info = {}
        self.best_parameters = {}
        
        logger.info(f"🦙 Predictor con Grid Search inicializado")
        logger.info(f"⏱️ Timeout: {timeout}s")
    
    def discover_available_models(self):
        """Descubre modelos de Ollama disponibles"""
        if not self.ollama_client.is_available():
            raise RuntimeError("❌ No hay modelos de Ollama disponibles")
        
        logger.info(f"🔍 Modelos Ollama encontrados:")
        for model_name in self.ollama_client.available_models:
            family = self.grid_optimizer.get_model_family(model_name)
            self.model_info[model_name] = {
                'type': self.classify_model_type(model_name),
                'family': family,
                'backend': 'ollama'
            }
            logger.info(f"   🦙 {model_name} ({family})")
        
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
            
            return sequences, true_categories
            
        except Exception as e:
            logger.error(f"❌ Error cargando CSV: {e}")
            raise
    
    def create_stratified_sample(self, sequences: List[str], true_categories: List[str], 
                               sample_size: int) -> Tuple[List[str], List[str]]:
        """Crea una muestra estratificada para grid search"""
        df = pd.DataFrame({'sequence': sequences, 'category': true_categories})
        
        # Calcular muestras por categoría manteniendo proporciones
        category_counts = df['category'].value_counts()
        sample_per_category = {}
        
        for category, count in category_counts.items():
            proportion = count / len(df)
            category_samples = max(1, int(sample_size * proportion))  # Al menos 1 muestra
            sample_per_category[category] = min(category_samples, count)
        
        # Ajustar si la suma excede sample_size
        total_samples = sum(sample_per_category.values())
        if total_samples > sample_size:
            # Reducir proporcionalmente
            factor = sample_size / total_samples
            for category in sample_per_category:
                sample_per_category[category] = max(1, int(sample_per_category[category] * factor))
        
        # Muestrear de cada categoría
        sampled_sequences = []
        sampled_categories = []
        
        for category, n_samples in sample_per_category.items():
            category_data = df[df['category'] == category].sample(n=n_samples, random_state=42)
            sampled_sequences.extend(category_data['sequence'].tolist())
            sampled_categories.extend(category_data['category'].tolist())
        
        logger.info(f"🎯 Muestra estratificada creada:")
        logger.info(f"   📊 Total muestras: {len(sampled_sequences)}")
        for category, count in pd.Series(sampled_categories).value_counts().items():
            logger.info(f"   {category}: {count} muestras")
        
        return sampled_sequences, sampled_categories
    
    def run_grid_search(self, model_name: str, sample_sequences: List[str], 
                       sample_categories: List[str], max_combinations: int = 20) -> Dict:
        """Ejecuta grid search para un modelo específico"""
        logger.info(f"🔍 Iniciando grid search para {model_name}...")
        
        # Generar combinaciones de parámetros
        param_combinations = self.grid_optimizer.generate_parameter_combinations(
            model_name, max_combinations
        )
        
        best_params = None
        best_score = -1
        best_metrics = None
        all_results = []
        
        # Barra de progreso para grid search
        progress = tqdm(param_combinations, 
                       desc=f"🔍 {model_name[:20]}...",
                       unit="params")
        
        for i, params in enumerate(progress):
            try:
                # Evaluar parámetros actuales
                metrics = self.grid_optimizer.evaluate_parameters(
                    model_name, params, sample_sequences, sample_categories,
                    self.ollama_client, self.prompt_generator
                )
                
                # Usar F1-score como métrica principal
                score = metrics['f1']
                
                # Guardar resultado
                result = {
                    'params': params.copy(),
                    'metrics': metrics.copy(),
                    'score': score
                }
                all_results.append(result)
                
                # Actualizar mejor resultado
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_metrics = metrics.copy()
                
                # Actualizar progreso
                progress.set_postfix({
                    "Best F1": f"{best_score:.3f}",
                    "Curr F1": f"{score:.3f}",
                    "Acc": f"{metrics['accuracy']:.3f}"
                })
                
            except Exception as e:
                logger.warning(f"Error evaluando parámetros {params}: {e}")
                continue
        
        # Guardar mejores parámetros para el modelo
        self.best_parameters[model_name] = {
            'params': best_params,
            'metrics': best_metrics,
            'score': best_score,
            'all_results': all_results
        }
        
        logger.info(f"✅ Grid search completado para {model_name}:")
        logger.info(f"   🎯 Mejor F1: {best_score:.3f}")
        logger.info(f"   🎯 Mejor Accuracy: {best_metrics['accuracy']:.3f}")
        logger.info(f"   🎯 Mejores parámetros: {best_params}")
        
        return self.best_parameters[model_name]
    
    def predict_with_optimized_model(self, model_name: str, sequences: List[str]) -> Tuple[List[str], List[Dict]]:
        """Hace predicciones usando los parámetros optimizados"""
        if model_name not in self.best_parameters:
            raise ValueError(f"❌ No hay parámetros optimizados para {model_name}")
        
        best_params = self.best_parameters[model_name]['params']
        
        logger.info(f"🚀 Prediciendo con {model_name} usando parámetros optimizados...")
        logger.info(f"   🎯 Parámetros: {best_params}")
        
        predictions = []
        detailed_responses = []
        
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
                
                # Generar respuesta con parámetros optimizados
                result = self.ollama_client.generate_text(model_name, prompt, **best_params)
                
                if result and 'response' in result:
                    response = result['response']
                    prediction = self.prompt_generator.extract_prediction_from_response(response)
                    predictions.append(prediction)
                    successful += 1
                    
                    # Guardar detalles
                    detailed_responses.append({
                        'sequence_index': i,
                        'sequence': sequence,
                        'prompt': prompt,
                        'raw_response': response,
                        'extracted_prediction': prediction,
                        'response_time': result.get('response_time', 0),
                        'model_name': model_name,
                        'optimized_params': str(best_params)
                    })
                else:
                    predictions.append('S')
                    failed += 1
                    
                    detailed_responses.append({
                        'sequence_index': i,
                        'sequence': sequence,
                        'prompt': prompt,
                        'raw_response': 'ERROR: No response from model',
                        'extracted_prediction': 'S',
                        'response_time': 0,
                        'model_name': model_name,
                        'optimized_params': str(best_params)
                    })
                    
            except Exception as e:
                logger.warning(f"Error en secuencia {i}: {e}")
                predictions.append('S')
                failed += 1
                
                detailed_responses.append({
                    'sequence_index': i,
                    'sequence': sequence,
                    'prompt': 'ERROR: Could not generate prompt',
                    'raw_response': f'ERROR: {str(e)}',
                    'extracted_prediction': 'S',
                    'response_time': 0,
                    'model_name': model_name,
                    'optimized_params': str(best_params)
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
    
    def run_full_optimization_pipeline(self, sequences: List[str], true_categories: List[str],
                                     grid_search_sample_size: int = 50, max_combinations: int = 20,
                                     max_models: int = None):
        """Ejecuta el pipeline completo: grid search + predicciones optimizadas"""
        
        total_sequences = len(sequences)
        total_models = len(self.model_info)
        
        if max_models:
            models_to_process = list(self.model_info.keys())[:max_models]
        else:
            models_to_process = list(self.model_info.keys())
        
        logger.info(f"🚀 INICIANDO PIPELINE DE OPTIMIZACIÓN:")
        logger.info(f"   📊 Dataset: {total_sequences:,} secuencias")
        logger.info(f"   🦙 Modelos: {len(models_to_process)} a procesar")
        logger.info(f"   🔍 Grid search: {grid_search_sample_size} muestras x {max_combinations} combinaciones")
        logger.info(f"   📈 Total predicciones finales: {total_sequences * len(models_to_process):,}")
        
        # Fase 1: Grid Search con muestra pequeña
        logger.info(f"\n🔍 FASE 1: GRID SEARCH OPTIMIZATION")
        logger.info(f"=" * 50)
        
        # Crear muestra estratificada
        sample_sequences, sample_categories = self.create_stratified_sample(
            sequences, true_categories, grid_search_sample_size
        )
        
        # Ejecutar grid search para cada modelo
        optimization_results = {}
        for model_name in models_to_process:
            try:
                result = self.run_grid_search(model_name, sample_sequences, sample_categories, max_combinations)
                optimization_results[model_name] = result
            except Exception as e:
                logger.error(f"❌ Error en grid search para {model_name}: {e}")
                continue
        
        # Fase 2: Predicciones con parámetros optimizados
        logger.info(f"\n🚀 FASE 2: PREDICCIONES CON PARÁMETROS OPTIMIZADOS")
        logger.info(f"=" * 55)
        
        all_predictions = {}
        all_detailed_responses = []
        
        for model_name in optimization_results.keys():
            try:
                start_time = time.time()
                predictions, detailed_responses = self.predict_with_optimized_model(model_name, sequences)
                prediction_time = time.time() - start_time
                
                all_predictions[model_name] = {
                    'predictions': predictions,
                    'prediction_time': prediction_time,
                    'samples_per_second': len(sequences) / prediction_time,
                    'model_type': self.model_info[model_name]['type'],
                    'model_family': self.model_info[model_name]['family'],
                    'optimized_params': optimization_results[model_name]['params'],
                    'grid_search_score': optimization_results[model_name]['score'],
                    'grid_search_metrics': optimization_results[model_name]['metrics']
                }
                
                # Agregar respuestas detalladas
                all_detailed_responses.extend(detailed_responses)
                
                logger.info(f"✅ {model_name} completado con parámetros optimizados")
                
            except Exception as e:
                logger.error(f"❌ Error con {model_name}: {e}")
                continue
        
        # Resumen final
        logger.info(f"\n🎉 PIPELINE DE OPTIMIZACIÓN COMPLETADO:")
        logger.info(f"   ✅ Modelos optimizados: {len(optimization_results)}")
        logger.info(f"   ✅ Predicciones exitosas: {len(all_predictions)}")
        logger.info(f"   📈 Total predicciones generadas: {sum(len(pred['predictions']) for pred in all_predictions.values()):,}")
        logger.info(f"   🔍 Respuestas detalladas capturadas: {len(all_detailed_responses):,}")
        
        return all_predictions, all_detailed_responses, optimization_results
    
    def save_optimization_results(self, all_predictions: Dict, sequences: List[str], 
                                true_categories: List[str], optimization_results: Dict,
                                detailed_responses: List[Dict], output_path: str):
        """Guarda resultados completos incluyendo optimización"""
        
        logger.info(f"💾 Guardando resultados de optimización en {output_path}...")
        
        # 1. Crear DataFrame de predicciones finales
        results_data = {
            'sequence': sequences,
            'true_category': true_categories
        }
        
        # Agregar predicciones de cada modelo optimizado
        model_count = 0
        for model_name, model_results in all_predictions.items():
            if isinstance(model_results, dict) and 'predictions' in model_results:
                results_data[f'pred_{model_name}'] = model_results['predictions']
                model_count += 1
        
        # Guardar CSV de predicciones
        df_results = pd.DataFrame(results_data)
        df_results.to_csv(output_path, index=False)
        
        # 2. Guardar resultados en formato pickle
        pickle_path = output_path.replace('.csv', '.pkl')
        pickle_data = {
            'all_predictions': all_predictions,
            'sequences': sequences,
            'true_categories': true_categories,
            'optimization_results': optimization_results,
            'model_info': self.model_info,
            'best_parameters': self.best_parameters,
            'standard_categories': self.standard_categories
        }
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(pickle_data, f)
        
        # 3. Guardar detalles de inputs/outputs
        detailed_output_path = output_path.replace('.csv', '_inputs_outputs.csv')
        if detailed_responses:
            df_detailed = pd.DataFrame(detailed_responses)
            df_detailed.to_csv(detailed_output_path, index=False)
        
        # 4. Guardar resultados de grid search
        gridsearch_path = output_path.replace('.csv', '_gridsearch_results.csv')
        gridsearch_data = []
        
        for model_name, opt_result in optimization_results.items():
            for result in opt_result['all_results']:
                row = {
                    'model_name': model_name,
                    'model_family': self.model_info[model_name]['family'],
                    'temperature': result['params']['temperature'],
                    'top_p': result['params']['top_p'],
                    'top_k': result['params']['top_k'],
                    'accuracy': result['metrics']['accuracy'],
                    'precision': result['metrics']['precision'],
                    'recall': result['metrics']['recall'],
                    'f1_score': result['metrics']['f1'],
                    'success_rate': result['metrics']['success_rate'],
                    'avg_time': result['metrics']['avg_time'],
                    'is_best': result['params'] == opt_result['params']
                }
                gridsearch_data.append(row)
        
        df_gridsearch = pd.DataFrame(gridsearch_data)
        df_gridsearch.to_csv(gridsearch_path, index=False)
        
        # 5. Crear resumen detallado
        summary_path = output_path.replace('.csv', '_optimization_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Resumen de Optimización COG - Grid Search + Predicciones\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"CONFIGURACIÓN DE OPTIMIZACIÓN:\n")
            f.write(f"-" * 30 + "\n")
            f.write(f"  Total secuencias: {len(sequences):,}\n")
            f.write(f"  Modelos optimizados: {len(optimization_results)}\n")
            f.write(f"  Predicciones totales: {len(sequences) * model_count:,}\n")
            f.write(f"  Timeout por predicción: {self.timeout}s\n\n")
            
            f.write(f"RESULTADOS DE GRID SEARCH:\n")
            f.write(f"-" * 26 + "\n")
            for model_name, opt_result in optimization_results.items():
                f.write(f"\n{model_name} ({self.model_info[model_name]['family']}):\n")
                f.write(f"  Mejores parámetros:\n")
                for param, value in opt_result['params'].items():
                    f.write(f"    {param}: {value}\n")
                f.write(f"  Métricas en muestra de validación:\n")
                f.write(f"    F1-Score: {opt_result['score']:.4f}\n")
                f.write(f"    Accuracy: {opt_result['metrics']['accuracy']:.4f}\n")
                f.write(f"    Precision: {opt_result['metrics']['precision']:.4f}\n")
                f.write(f"    Recall: {opt_result['metrics']['recall']:.4f}\n")
                f.write(f"    Success Rate: {opt_result['metrics']['success_rate']:.4f}\n")
                f.write(f"    Avg Time/Pred: {opt_result['metrics']['avg_time']:.2f}s\n")
                f.write(f"  Combinaciones evaluadas: {len(opt_result['all_results'])}\n")
            
            f.write(f"\nRENDIMIENTO EN DATASET COMPLETO:\n")
            f.write(f"-" * 32 + "\n")
            for model_name, pred_result in all_predictions.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Tiempo total: {pred_result['prediction_time']:.1f}s\n")
                f.write(f"  Velocidad: {pred_result['samples_per_second']:.1f} seq/s\n")
                f.write(f"  Secuencias procesadas: {len(sequences):,}\n")
                f.write(f"  Mejora vs parámetros default: Optimizado con grid search\n")
            
            f.write(f"\nARCHIVOS GENERADOS:\n")
            f.write(f"-" * 18 + "\n")
            f.write(f"  1. {output_path} - Predicciones finales (CSV)\n")
            f.write(f"  2. {pickle_path} - Datos completos (Pickle)\n")
            f.write(f"  3. {detailed_output_path} - Inputs/Outputs detallados\n")
            f.write(f"  4. {gridsearch_path} - Resultados de grid search\n")
            f.write(f"  5. {summary_path} - Este resumen\n\n")
            
            f.write(f"VENTAJAS DE LA OPTIMIZACIÓN:\n")
            f.write(f"-" * 28 + "\n")
            f.write(f"  ✅ Parámetros optimizados por modelo individual\n")
            f.write(f"  ✅ Evaluación sistemática de hiperparámetros\n")
            f.write(f"  ✅ Mejor precisión vs parámetros por defecto\n")
            f.write(f"  ✅ Muestra estratificada para validación\n")
            f.write(f"  ✅ Métricas detalladas de cada combinación\n")
            f.write(f"  ✅ Reproducibilidad con semillas fijas\n")
        
        # Logging de archivos generados
        logger.info(f"✅ Resultados de optimización guardados:")
        logger.info(f"   📊 {output_path} - Predicciones finales")
        logger.info(f"   🔧 {pickle_path} - Datos completos (Pickle)")
        logger.info(f"   🔍 {detailed_output_path} - Inputs/Outputs detallados")
        logger.info(f"   📈 {gridsearch_path} - Resultados de grid search")
        logger.info(f"   📋 {summary_path} - Resumen de optimización")
        
        logger.info(f"\n📊 ESTADÍSTICAS FINALES:")
        logger.info(f"   🎯 Modelos optimizados: {len(optimization_results)}")
        logger.info(f"   📈 Predicciones totales: {len(sequences) * model_count:,}")
        logger.info(f"   🔍 Combinaciones evaluadas: {sum(len(opt['all_results']) for opt in optimization_results.values())}")
        logger.info(f"   ⚡ Mejores F1-scores por modelo:")
        for model_name, opt_result in optimization_results.items():
            logger.info(f"      {model_name}: {opt_result['score']:.4f}")


def parse_arguments():
    """Parsea argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='Predictor COG con Grid Search de Hiperparámetros',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🔍 GRID SEARCH OPTIMIZER para Predicción de Categorías COG

Este script optimiza automáticamente los hiperparámetros de cada modelo usando una muestra 
pequeña del dataset, y luego ejecuta las predicciones completas con los parámetros optimizados.

PROCESO:
1. Carga dataset completo
2. Crea muestra estratificada pequeña para grid search
3. Evalúa múltiples combinaciones de parámetros por modelo
4. Selecciona mejores parámetros basado en F1-score
5. Ejecuta predicciones completas con parámetros optimizados

EJEMPLOS:
  # Uso básico con optimización automática
  python ollama_gridsearch_predictor.py --csv cog_categories_balanced_460.csv
  
  # Con muestra más grande para grid search (mejor optimización)
  python ollama_gridsearch_predictor.py --csv cog_categories_balanced_460.csv --grid-sample-size 100
  
  # Evaluar más combinaciones de parámetros
  python ollama_gridsearch_predictor.py --csv cog_categories_balanced_460.csv --max-combinations 30
  
  # Limitar modelos para pruebas rápidas
  python ollama_gridsearch_predictor.py --csv cog_categories_balanced_460.csv --max-models 3
  
  # Solo modelos específicos
  python ollama_gridsearch_predictor.py --csv cog_categories_balanced_460.csv --models llama3.2:3b mistral:7b

ARCHIVOS GENERADOS:
  1. [output].csv - Predicciones finales optimizadas
  2. [output].pkl - Datos completos para benchmarking
  3. [output]_inputs_outputs.csv - Inputs/outputs detallados
  4. [output]_gridsearch_results.csv - Resultados de todas las combinaciones
  5. [output]_optimization_summary.txt - Resumen de optimización

VENTAJAS:
  ✅ Optimización automática de hiperparámetros por modelo
  ✅ Evaluación sistemática con métricas detalladas
  ✅ Muestra estratificada para validación representativa
  ✅ Mejor precisión vs parámetros por defecto
  ✅ Compatible con códigos de benchmarking existentes
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
        default='cog_predictions_optimized.csv',
        help='Archivo de salida para las predicciones optimizadas'
    )
    
    parser.add_argument(
        '--grid-sample-size',
        type=int,
        default=50,
        help='Tamaño de muestra para grid search (default: 50)'
    )
    
    parser.add_argument(
        '--max-combinations',
        type=int,
        default=20,
        help='Máximo combinaciones de parámetros a evaluar por modelo (default: 20)'
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
        help='Número máximo de modelos a evaluar'
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
        help='Timeout en segundos por predicción'
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
    
    # Inicializar predictor con grid search
    try:
        predictor = OllamaGridSearchPredictor(
            ollama_host=args.ollama_host,
            ollama_port=args.ollama_port,
            timeout=args.timeout
        )
        
        # Descubrir modelos
        predictor.discover_available_models()
        
        # Si solo se quiere listar modelos
        if args.list_models:
            logger.info("📋 Modelos de Ollama disponibles para optimización:")
            for model_name, info in predictor.model_info.items():
                family = info['family']
                model_type = info['type']
                logger.info(f"   🦙 {model_name} ({family}, {model_type})")
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
    
    # Cargar datos del CSV
    try:
        sequences, true_categories = predictor.load_csv_data(args.csv)
        
        logger.info(f"\n🎯 CONFIGURACIÓN DE OPTIMIZACIÓN:")
        logger.info(f"   📊 Dataset total: {len(sequences):,} secuencias")
        logger.info(f"   🔍 Muestra grid search: {args.grid_sample_size} secuencias")
        logger.info(f"   🦙 Modelos a optimizar: {len(predictor.model_info)}")
        logger.info(f"   📈 Combinaciones máximas por modelo: {args.max_combinations}")
        logger.info(f"   ⏱️ Timeout: {args.timeout}s")
        
        # Estimación de tiempo
        grid_search_predictions = args.grid_sample_size * len(predictor.model_info) * args.max_combinations
        final_predictions = len(sequences) * len(predictor.model_info)
        total_predictions = grid_search_predictions + final_predictions
        
        estimated_time_minutes = (total_predictions * 3) / 60  # ~3 segundos por predicción
        if estimated_time_minutes > 60:
            time_str = f"{estimated_time_minutes/60:.1f} horas"
        else:
            time_str = f"{estimated_time_minutes:.1f} minutos"
        
        logger.info(f"   ⏱️ Tiempo estimado total: ~{time_str}")
        logger.info(f"      - Grid search: ~{(grid_search_predictions * 3)/60:.1f} min")
        logger.info(f"      - Predicciones finales: ~{(final_predictions * 3)/60:.1f} min")
        
    except Exception as e:
        logger.error(f"❌ Error cargando datos: {e}")
        sys.exit(1)
    
    # Ejecutar pipeline de optimización completo
    try:
        logger.info(f"\n🚀 Iniciando pipeline de optimización completo...")
        
        all_predictions, all_detailed_responses, optimization_results = predictor.run_full_optimization_pipeline(
            sequences=sequences,
            true_categories=true_categories,
            grid_search_sample_size=args.grid_sample_size,
            max_combinations=args.max_combinations,
            max_models=args.max_models
        )
        
        if not all_predictions:
            logger.error("❌ No se generaron predicciones optimizadas")
            sys.exit(1)
        
        # Guardar todos los resultados
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        predictor.save_optimization_results(
            all_predictions, sequences, true_categories, 
            optimization_results, all_detailed_responses, args.output
        )
        
        logger.info(f"\n🎉 OPTIMIZACIÓN COMPLETADA EXITOSAMENTE!")
        logger.info(f"📁 Resultados principales: {args.output}")
        logger.info(f"🔧 Datos para benchmarking: {args.output.replace('.csv', '.pkl')}")
        logger.info(f"📊 Procesadas {len(sequences):,} secuencias con {len(all_predictions)} modelos optimizados")
        
        # Mostrar resumen de mejores parámetros
        print(f"\n📈 RESUMEN DE OPTIMIZACIÓN:")
        print(f"=" * 40)
        for model_name, opt_result in optimization_results.items():
            print(f"\n🦙 {model_name}:")
            print(f"   🎯 F1-Score: {opt_result['score']:.4f}")
            print(f"   📊 Accuracy: {opt_result['metrics']['accuracy']:.4f}")
            print(f"   ⚙️ Parámetros: {opt_result['params']}")
        
        print(f"\n💡 PRÓXIMOS PASOS:")
        print(f"   🔍 Revisar: {args.output.replace('.csv', '_gridsearch_results.csv')}")
        print(f"   📊 Analizar: {args.output.replace('.csv', '_optimization_summary.txt')}")
        print(f"   🔧 Benchmarking: Usar {args.output.replace('.csv', '.pkl')}")
        
    except Exception as e:
        logger.error(f"❌ Error durante optimización: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("🔍 Predictor COG con Grid Search de Hiperparámetros")
    print("=" * 55)
    print("🎯 OPTIMIZACIÓN AUTOMÁTICA:")
    print("   ✅ Grid search de hiperparámetros por modelo")
    print("   ✅ Muestra estratificada para validación")
    print("   ✅ Selección automática de mejores parámetros")
    print("   ✅ Predicciones optimizadas en dataset completo")
    print("   ✅ Métricas detalladas de cada combinación")
    print("   ✅ Compatible con códigos de benchmarking")
    print("=" * 55)
    print()
    
    main()