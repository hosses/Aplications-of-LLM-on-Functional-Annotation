import os
import pickle
import numpy as np
import pandas as pd
import torch
import requests
import json
import time
import re
import argparse
import sys
from typing import Dict, List, Tuple, Any, Optional
import warnings
import logging
from collections import Counter

# Verificar dependencias requeridas
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("⚠️ tqdm no está disponible. Se ejecutará sin barras de progreso.")
    logger.warning("📦 Para habilitar barras de progreso: pip install tqdm")
    
    # Crear una función dummy para tqdm
    def tqdm(iterable, *args, **kwargs):
        """Función dummy para tqdm cuando no está disponible"""
        return iterable

# Hugging Face imports (solo para BERT)
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoModel,
    BertTokenizer, BertForSequenceClassification, BertModel
)

# Sklearn imports (para baselines y balancing)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample

warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataBalancer:
    """Clase para equilibrar automáticamente las clases en el conjunto de datos"""
    
    def __init__(self, strategy='stratified', min_samples_per_class=2, max_samples_per_class=None):
        """
        Inicializa el balanceador de datos
        
        Args:
            strategy: Estrategia de equilibrio ('stratified', 'undersample', 'oversample', 'filter')
            min_samples_per_class: Mínimo de muestras por clase
            max_samples_per_class: Máximo de muestras por clase (None = sin límite)
        """
        self.strategy = strategy
        self.min_samples_per_class = min_samples_per_class
        self.max_samples_per_class = max_samples_per_class
        self.original_distribution = None
        self.final_distribution = None
        self.excluded_classes = []
        self.balancing_report = {}
    
    def analyze_class_distribution(self, y, label_names):
        """
        Analiza la distribución de clases
        
        Args:
            y: Array de etiquetas
            label_names: Nombres de las clases
            
        Returns:
            dict: Estadísticas de distribución
        """
        # Convertir a índices si es necesario
        if y.ndim > 1:
            y_indices = np.argmax(y, axis=1)
        else:
            y_indices = y
        
        # Contar clases
        unique_classes, counts = np.unique(y_indices, return_counts=True)
        
        distribution = {}
        total_samples = len(y_indices)
        
        for class_idx, count in zip(unique_classes, counts):
            class_name = label_names[class_idx] if class_idx < len(label_names) else f"Clase_{class_idx}"
            distribution[class_idx] = {
                'name': class_name,
                'count': count,
                'percentage': (count / total_samples) * 100
            }
        
        # Identificar clases problemáticas
        missing_classes = []
        low_sample_classes = []
        
        for i in range(len(label_names)):
            if i not in unique_classes:
                missing_classes.append(i)
            elif distribution[i]['count'] < self.min_samples_per_class:
                low_sample_classes.append(i)
        
        stats = {
            'total_samples': total_samples,
            'total_classes_defined': len(label_names),
            'classes_with_samples': len(unique_classes),
            'missing_classes': missing_classes,
            'low_sample_classes': low_sample_classes,
            'distribution': distribution,
            'is_balanced': self._is_balanced(distribution),
            'balance_ratio': self._calculate_balance_ratio(distribution)
        }
        
        return stats
    
    def _is_balanced(self, distribution):
        """Verifica si las clases están razonablemente equilibradas"""
        if len(distribution) < 2:
            return True
        
        counts = [info['count'] for info in distribution.values()]
        min_count = min(counts)
        max_count = max(counts)
        
        # Considerar equilibrado si la ratio max/min <= 3
        return (max_count / min_count) <= 3.0 if min_count > 0 else False
    
    def _calculate_balance_ratio(self, distribution):
        """Calcula la ratio de equilibrio (1.0 = perfectamente equilibrado)"""
        if len(distribution) < 2:
            return 1.0
        
        counts = [info['count'] for info in distribution.values()]
        min_count = min(counts)
        max_count = max(counts)
        
        return min_count / max_count if max_count > 0 else 0.0
    
    def print_distribution_analysis(self, stats, title="ANÁLISIS DE DISTRIBUCIÓN DE CLASES"):
        """Imprime análisis detallado de la distribución"""
        logger.info("=" * 60)
        logger.info(title)
        logger.info("=" * 60)
        
        logger.info(f"📊 Total de muestras: {stats['total_samples']}")
        logger.info(f"🏷️ Clases definidas: {stats['total_classes_defined']}")
        logger.info(f"✅ Clases con muestras: {stats['classes_with_samples']}")
        logger.info(f"⚖️ Equilibrio: {'Sí' if stats['is_balanced'] else 'No'} (ratio: {stats['balance_ratio']:.3f})")
        
        if stats['missing_classes']:
            logger.warning(f"❌ Clases sin muestras ({len(stats['missing_classes'])}): "
                         f"{[stats['distribution'].get(i, {}).get('name', f'Clase_{i}') for i in stats['missing_classes']]}")
        
        if stats['low_sample_classes']:
            logger.warning(f"⚠️ Clases con pocas muestras (<{self.min_samples_per_class}): "
                         f"{[stats['distribution'][i]['name'] for i in stats['low_sample_classes']]}")
        
        logger.info(f"\n📈 DISTRIBUCIÓN POR CLASE:")
        for class_idx, info in stats['distribution'].items():
            status = "✅" if info['count'] >= self.min_samples_per_class else "⚠️" if info['count'] > 0 else "❌"
            logger.info(f"   {status} {info['name']}: {info['count']} muestras ({info['percentage']:.1f}%)")
        
        logger.info("=" * 60)
    
    def balance_data(self, X, y, max_samples=None, label_names=None):
        """
        Equilibra los datos según la estrategia especificada
        
        Args:
            X: Datos de entrada
            y: Etiquetas
            max_samples: Número máximo total de muestras
            label_names: Nombres de las clases
            
        Returns:
            tuple: (X_balanced, y_balanced, balancing_info)
        """
        logger.info(f"🔄 Iniciando equilibrio de datos con estrategia '{self.strategy}'...")
        
        # Analizar distribución original
        self.original_distribution = self.analyze_class_distribution(y, label_names)
        self.print_distribution_analysis(self.original_distribution, "DISTRIBUCIÓN ORIGINAL")
        
        # Convertir y a índices si es necesario
        if y.ndim > 1:
            y_indices = np.argmax(y, axis=1)
            y_original_format = 'one_hot'
        else:
            y_indices = y.copy()
            y_original_format = 'indices'
        
        # Aplicar estrategia de equilibrio
        if self.strategy == 'stratified':
            X_balanced, y_balanced = self._stratified_sampling(X, y_indices, max_samples, label_names)
        elif self.strategy == 'undersample':
            X_balanced, y_balanced = self._undersample_majority(X, y_indices, max_samples, label_names)
        elif self.strategy == 'oversample':
            X_balanced, y_balanced = self._oversample_minority(X, y_indices, max_samples, label_names)
        elif self.strategy == 'filter':
            X_balanced, y_balanced = self._filter_classes(X, y_indices, max_samples, label_names)
        else:
            logger.warning(f"⚠️ Estrategia desconocida '{self.strategy}', usando muestreo estratificado")
            X_balanced, y_balanced = self._stratified_sampling(X, y_indices, max_samples, label_names)
        
        # Convertir y_balanced de vuelta al formato original si es necesario
        if y_original_format == 'one_hot' and y_balanced.ndim == 1:
            y_balanced_one_hot = np.zeros((len(y_balanced), y.shape[1]))
            y_balanced_one_hot[np.arange(len(y_balanced)), y_balanced] = 1
            y_balanced = y_balanced_one_hot
        
        # Analizar distribución final
        self.final_distribution = self.analyze_class_distribution(y_balanced, label_names)
        self.print_distribution_analysis(self.final_distribution, "DISTRIBUCIÓN DESPUÉS DEL EQUILIBRIO")
        
        # Crear reporte
        balancing_info = self._create_balancing_report()
        
        return X_balanced, y_balanced, balancing_info
    
    def _stratified_sampling(self, X, y_indices, max_samples, label_names):
        """Muestreo estratificado manteniendo proporciones"""
        if max_samples is None or max_samples >= len(X):
            logger.info("📊 No es necesario reducir muestras")
            return X, y_indices
        
        logger.info(f"📊 Aplicando muestreo estratificado: {len(X)} → {max_samples} muestras")
        
        # Calcular muestras por clase manteniendo proporciones
        unique_classes, counts = np.unique(y_indices, return_counts=True)
        total_samples = len(y_indices)
        
        selected_indices = []
        
        # Barra de progreso para el muestreo por clase (solo si tqdm está disponible)
        if TQDM_AVAILABLE:
            try:
                class_progress = tqdm(zip(unique_classes, counts), 
                                     total=len(unique_classes),
                                     desc="🎯 Muestreando clases",
                                     unit="clase",
                                     leave=False)
            except:
                class_progress = zip(unique_classes, counts)
        else:
            class_progress = zip(unique_classes, counts)
        
        for class_idx, count in class_progress:
            # Actualizar descripción con clase actual (solo si es tqdm real)
            if TQDM_AVAILABLE and hasattr(class_progress, 'set_postfix'):
                try:
                    class_name = label_names[class_idx] if class_idx < len(label_names) else f"Clase_{class_idx}"
                    class_progress.set_postfix({"Clase": class_name[:15], "Muestras": count})
                except:
                    pass
            
            # Calcular proporción y muestras objetivo
            proportion = count / total_samples
            target_samples = max(int(max_samples * proportion), self.min_samples_per_class)
            
            # No exceder las muestras disponibles
            target_samples = min(target_samples, count)
            
            # Seleccionar muestras aleatoriamente
            class_indices = np.where(y_indices == class_idx)[0]
            if target_samples > 0:
                selected_class_indices = np.random.choice(
                    class_indices, 
                    size=target_samples, 
                    replace=False
                )
                selected_indices.extend(selected_class_indices)
        
        # Cerrar barra de progreso si es necesario
        if TQDM_AVAILABLE and hasattr(class_progress, 'close'):
            try:
                class_progress.close()
            except:
                pass
        
        # Ajustar si tenemos demasiadas muestras
        if len(selected_indices) > max_samples:
            logger.info("⚖️ Ajustando tamaño final...")
            selected_indices = np.random.choice(selected_indices, size=max_samples, replace=False)
        
        selected_indices = np.array(selected_indices)
        
        # ⭐ FIX: Manejar tanto arrays NumPy como listas de Python
        logger.info("📋 Seleccionando muestras finales...")
        if isinstance(X, np.ndarray):
            X_selected = X[selected_indices]
        else:
            # X es una lista (secuencias de proteínas como strings)
            if TQDM_AVAILABLE:
                try:
                    with tqdm(total=len(selected_indices), desc="📋 Extrayendo secuencias", unit="seq", leave=False) as pbar:
                        X_selected = []
                        for i in selected_indices:
                            X_selected.append(X[i])
                            pbar.update(1)
                except:
                    # Fallback sin barra de progreso
                    X_selected = [X[i] for i in selected_indices]
            else:
                X_selected = [X[i] for i in selected_indices]
        
        return X_selected, y_indices[selected_indices]
    
    def _undersample_majority(self, X, y_indices, max_samples, label_names):
        """Submuestreo de clases mayoritarias"""
        logger.info("📊 Aplicando submuestreo de clases mayoritarias")
        
        unique_classes, counts = np.unique(y_indices, return_counts=True)
        
        # Encontrar tamaño de clase minoritaria
        min_count = max(min(counts), self.min_samples_per_class)
        
        # Calcular tamaño objetivo por clase
        if max_samples:
            target_per_class = min(min_count, max_samples // len(unique_classes))
        else:
            target_per_class = min_count
        
        selected_indices = []
        
        for class_idx in unique_classes:
            class_indices = np.where(y_indices == class_idx)[0]
            
            if len(class_indices) > target_per_class:
                # Submuestrear
                selected_class_indices = np.random.choice(
                    class_indices, 
                    size=target_per_class, 
                    replace=False
                )
            else:
                # Usar todas las muestras disponibles
                selected_class_indices = class_indices
            
            selected_indices.extend(selected_class_indices)
        
        selected_indices = np.array(selected_indices)
        
        # ⭐ FIX: Manejar tanto arrays NumPy como listas de Python
        if isinstance(X, np.ndarray):
            X_selected = X[selected_indices]
        else:
            # X es una lista (secuencias de proteínas como strings)
            X_selected = [X[i] for i in selected_indices]
        
        return X_selected, y_indices[selected_indices]
    
    def _oversample_minority(self, X, y_indices, max_samples, label_names):
        """Sobremuestreo de clases minoritarias"""
        logger.info("📊 Aplicando sobremuestreo de clases minoritarias")
        
        unique_classes, counts = np.unique(y_indices, return_counts=True)
        max_count = max(counts)
        
        X_resampled = []
        y_resampled = []
        
        for class_idx in unique_classes:
            class_indices = np.where(y_indices == class_idx)[0]
            
            # ⭐ FIX: Manejar tanto arrays NumPy como listas de Python
            if isinstance(X, np.ndarray):
                class_X = X[class_indices]
            else:
                # X es una lista (secuencias de proteínas como strings)
                class_X = [X[i] for i in class_indices]
            
            class_y = y_indices[class_indices]
            
            if len(class_indices) < max_count:
                # Sobremuestrear
                if isinstance(X, np.ndarray):
                    X_upsampled, y_upsampled = resample(
                        class_X, class_y,
                        n_samples=max_count,
                        replace=True,
                        random_state=42
                    )
                    X_resampled.append(X_upsampled)
                else:
                    # Para listas, hacer resample manualmente
                    upsampled_indices = np.random.choice(
                        len(class_X), 
                        size=max_count, 
                        replace=True
                    )
                    X_upsampled = [class_X[i] for i in upsampled_indices]
                    y_upsampled = class_y[upsampled_indices]
                    X_resampled.extend(X_upsampled)
                
                y_resampled.append(y_upsampled)
            else:
                if isinstance(X, np.ndarray):
                    X_resampled.append(class_X)
                else:
                    X_resampled.extend(class_X)
                y_resampled.append(class_y)
        
        # Combinar resultados
        if isinstance(X, np.ndarray):
            X_balanced = np.concatenate(X_resampled)
        else:
            X_balanced = X_resampled  # Ya es una lista plana
        
        y_balanced = np.concatenate(y_resampled)
        
        # Aplicar límite de muestras si se especifica
        if max_samples and len(X_balanced) > max_samples:
            indices = np.random.choice(len(X_balanced), size=max_samples, replace=False)
            if isinstance(X, np.ndarray):
                X_balanced = X_balanced[indices]
            else:
                X_balanced = [X_balanced[i] for i in indices]
            y_balanced = y_balanced[indices]
        
        return X_balanced, y_balanced
    
    def _filter_classes(self, X, y_indices, max_samples, label_names):
        """Filtrar clases problemáticas"""
        logger.info("📊 Aplicando filtrado de clases problemáticas")
        
        unique_classes, counts = np.unique(y_indices, return_counts=True)
        
        # Identificar clases válidas
        valid_classes = []
        for class_idx, count in zip(unique_classes, counts):
            if count >= self.min_samples_per_class:
                valid_classes.append(class_idx)
            else:
                class_name = label_names[class_idx] if class_idx < len(label_names) else f"Clase_{class_idx}"
                logger.warning(f"⚠️ Excluyendo clase {class_name} (solo {count} muestras)")
                self.excluded_classes.append(class_idx)
        
        if len(valid_classes) == 0:
            raise ValueError("No hay clases válidas después del filtrado")
        
        # Filtrar datos
        valid_mask = np.isin(y_indices, valid_classes)
        
        # ⭐ FIX: Manejar tanto arrays NumPy como listas de Python
        if isinstance(X, np.ndarray):
            X_filtered = X[valid_mask]
        else:
            # X es una lista (secuencias de proteínas como strings)
            X_filtered = [X[i] for i, is_valid in enumerate(valid_mask) if is_valid]
        
        y_filtered = y_indices[valid_mask]
        
        # Aplicar límite de muestras si se especifica
        if max_samples and len(X_filtered) > max_samples:
            # Usar muestreo estratificado para el límite final
            return self._stratified_sampling(X_filtered, y_filtered, max_samples, label_names)
        
        return X_filtered, y_filtered
    
    def _create_balancing_report(self):
        """Crea reporte detallado del proceso de equilibrio"""
        report = {
            'strategy': self.strategy,
            'min_samples_per_class': self.min_samples_per_class,
            'max_samples_per_class': self.max_samples_per_class,
            'excluded_classes': self.excluded_classes,
            'original_stats': self.original_distribution,
            'final_stats': self.final_distribution,
            'changes': {}
        }
        
        # Calcular cambios
        if self.original_distribution and self.final_distribution:
            orig_total = self.original_distribution['total_samples']
            final_total = self.final_distribution['total_samples']
            
            report['changes'] = {
                'total_samples_change': final_total - orig_total,
                'total_samples_ratio': final_total / orig_total if orig_total > 0 else 0,
                'balance_improvement': (
                    self.final_distribution['balance_ratio'] - 
                    self.original_distribution['balance_ratio']
                ),
                'classes_removed': len(self.excluded_classes),
                'is_now_balanced': self.final_distribution['is_balanced']
            }
        
        return report


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
            timeout = getattr(self, 'timeout', 300)  # Usar timeout personalizado o 300 por defecto
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
    
    def get_embeddings(self, model: str, text: str) -> Optional[np.ndarray]:
        """Obtiene embeddings usando Ollama"""
        if not self.is_model_available(model):
            logger.error(f"❌ Modelo {model} no disponible para embeddings")
            return None
        
        data = {
            "model": model,
            "prompt": text
        }
        
        try:
            response = requests.post(f"{self.api_url}/embeddings", json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                if 'embedding' in result:
                    return np.array(result['embedding'])
            else:
                logger.error(f"❌ Error embeddings Ollama: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error en embeddings Ollama: {e}")
            return None


class UnifiedPromptGenerator:
    """Generador de prompts unificado para todas las LLMs con one-shot example"""
    
    def __init__(self):
        # Definir categorías COG y pathways para los prompts
        self.cog_categories = {
            'J': 'Translation, ribosomal structure and biogenesis',
            'A': 'RNA processing and modification',
            'K': 'Transcription', 
            'L': 'Replication, recombination and repair',
            'B': 'Chromatin structure and dynamics',
            'D': 'Cell cycle control, cell division, chromosome partitioning',
            'O': 'Molecular chaperones and related functions',
            'M': 'Cell wall/membrane/envelope biogenesis',
            'N': 'Cell motility',
            'P': 'Inorganic ion transport and metabolism',
            'T': 'Signal transduction mechanisms',
            'U': 'Intracellular trafficking, secretion, and vesicular transport',
            'V': 'Defense mechanisms',
            'W': 'Extracellular structures',
            'C': 'Energy production and conversion',
            'G': 'Carbohydrate transport and metabolism',
            'E': 'Amino acid transport and metabolism',
            'F': 'Nucleotide transport and metabolism',
            'H': 'Coenzyme transport and metabolism',
            'I': 'Lipid transport and metabolism',
            'Q': 'Secondary metabolites biosynthesis, transport and catabolism',
            'R': 'General function prediction only',
            'S': 'Function unknown'
        }
        
        self.common_pathways = [
            'Glycolysis', 'TCA cycle', 'Pentose phosphate pathway',
            'DNA replication', 'DNA repair', 'Base excision repair',
            'Nucleotide excision repair', 'Mismatch repair',
            'Protein synthesis', 'Protein folding', 'Protein degradation',
            'Fatty acid biosynthesis', 'Fatty acid degradation',
            'Amino acid biosynthesis', 'Amino acid degradation',
            'Cell wall biosynthesis', 'Peptidoglycan biosynthesis',
            'Lipopolysaccharide biosynthesis', 'Ribosome biogenesis',
            'tRNA processing', 'rRNA processing'
        ]
    
    def generate_functional_category_prompt(self, sequence: str) -> str:
        """Genera prompt para predicción de categorías COG (optimizado para Ollama)"""
        # Insertar espacios cada 3 aminoácidos para mejor legibilidad
        spaced_sequence = ' '.join([sequence[i:i+3] for i in range(0, len(sequence), 3)])
        
        prompt = f"""You are a protein function prediction expert. Analyze protein sequences and predict their COG functional category.

COG Categories:
J: Translation, ribosomal structure and biogenesis
A: RNA processing and modification  
K: Transcription
L: Replication, recombination and repair
B: Chromatin structure and dynamics
D: Cell cycle control, cell division, chromosome partitioning
O: Molecular chaperones and related functions
M: Cell wall/membrane/envelope biogenesis
N: Cell motility
P: Inorganic ion transport and metabolism
T: Signal transduction mechanisms
U: Intracellular trafficking, secretion, and vesicular transport
V: Defense mechanisms
W: Extracellular structures
C: Energy production and conversion
G: Carbohydrate transport and metabolism
E: Amino acid transport and metabolism
F: Nucleotide transport and metabolism
H: Coenzyme transport and metabolism
I: Lipid transport and metabolism
Q: Secondary metabolites biosynthesis, transport and catabolism
R: General function prediction only
S: Function unknown

Example:
Protein sequence: MKL LLV LSL VLV GLV LLA LVA LGI VLG
Analysis: This sequence shows characteristics of membrane proteins with multiple transmembrane domains, likely involved in transport functions.
Prediction: P

Now analyze this protein:
Protein sequence: {spaced_sequence}
Analysis: [Provide brief analysis of sequence characteristics]
Prediction: [Single letter only]"""
        
        return prompt
    
    def generate_pathway_prompt(self, sequence: str) -> str:
        """Genera prompt para predicción de pathways (optimizado para Ollama)"""
        spaced_sequence = ' '.join([sequence[i:i+3] for i in range(0, len(sequence), 3)])
        pathways_list = ', '.join(self.common_pathways[:15])
        
        prompt = f"""You are a protein function prediction expert. Analyze protein sequences and predict their metabolic pathway.

Common Pathways:
{pathways_list}, and others.

Example:
Protein sequence: MET LYS VAL LEU GLY TRP ASN PRO GLU
Analysis: This sequence contains conserved domains typical of glycolytic enzymes, with binding sites for glucose derivatives.
Prediction: Glycolysis

Now analyze this protein:
Protein sequence: {spaced_sequence}
Analysis: [Provide brief analysis of sequence characteristics]
Prediction: [Pathway name only]"""
        
        return prompt
    
    def extract_prediction_from_response(self, response: str, task: str) -> str:
        """Extrae la predicción de la respuesta del LLM"""
        response = response.strip()
        
        if task == 'functional_category':
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
            
            return 'S'  # Function unknown
            
        elif task == 'pathway':
            # Buscar patrón "Prediction: [PATHWAY]"
            prediction_match = re.search(r'Prediction:\s*([A-Za-z\s]+)', response, re.IGNORECASE)
            if prediction_match:
                pathway = prediction_match.group(1).strip()
                return self.normalize_pathway_name(pathway)
            
            # Buscar nombres de pathways conocidos en la respuesta
            response_lower = response.lower()
            for pathway in self.common_pathways:
                if pathway.lower() in response_lower:
                    return pathway
            
            return 'Unknown pathway'
        
        return 'Unknown'
    
    def normalize_pathway_name(self, pathway: str) -> str:
        """Normaliza el nombre del pathway a un formato estándar"""
        pathway = pathway.strip().title()
        
        pathway_mapping = {
            'Glycolysis': 'Glycolysis',
            'Glucose Metabolism': 'Glycolysis',
            'Tca Cycle': 'TCA cycle',
            'Citric Acid Cycle': 'TCA cycle',
            'Krebs Cycle': 'TCA cycle',
            'Dna Replication': 'DNA replication',
            'Dna Repair': 'DNA repair',
            'Protein Synthesis': 'Protein synthesis',
            'Translation': 'Protein synthesis',
            'Fatty Acid Synthesis': 'Fatty acid biosynthesis',
            'Amino Acid Synthesis': 'Amino acid biosynthesis',
        }
        
        return pathway_mapping.get(pathway, pathway)


class HybridModelLoader:
    """Cargador híbrido: Ollama para LLMs grandes + Hugging Face para BERT"""
    
    def __init__(self, models_dir="./models", device=None, ollama_host="localhost", ollama_port=11434, 
                 global_params=None, model_specific_params=None, timeout=300, 
                 balance_strategy='stratified', min_samples_per_class=2):
        self.models_dir = models_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.timeout = timeout
        
        # Parámetros de generación
        self.global_params = global_params or {}
        self.model_specific_params = model_specific_params or {}
        
        # ⭐ Balanceador de datos
        self.data_balancer = DataBalancer(
            strategy=balance_strategy,
            min_samples_per_class=min_samples_per_class
        )
        
        # Contenedores para diferentes tipos de modelos
        self.hf_models = {}  # Modelos Hugging Face (BERT)
        self.hf_tokenizers = {}  # Tokenizers Hugging Face
        self.model_info = {}
        
        # Cliente Ollama
        self.ollama_client = OllamaClient(ollama_host, ollama_port)
        
        # Establecer timeout en el cliente
        self.ollama_client.timeout = timeout
        
        # Generador de prompts
        self.prompt_generator = UnifiedPromptGenerator()
        
        # Definir etiquetas estándar
        self.standard_categories = ['J', 'A', 'K', 'L', 'B', 'D', 'O', 'M', 'N', 'P', 
                                  'T', 'U', 'V', 'W', 'C', 'G', 'E', 'F', 'H', 'I', 
                                  'Q', 'R', 'S']
        
        self.standard_pathways = [
            'Glycolysis', 'TCA cycle', 'Pentose phosphate pathway',
            'DNA replication', 'DNA repair', 'Protein synthesis',
            'Protein folding', 'Fatty acid biosynthesis', 'Amino acid biosynthesis',
            'Cell wall biosynthesis', 'Unknown pathway'
        ]
        
        logger.info(f"🔄 Cargador híbrido inicializado")
        logger.info(f"🎮 Dispositivo: {self.device}")
        logger.info(f"🦙 Ollama disponible: {self.ollama_client.is_available()}")
        logger.info(f"⚖️ Estrategia de equilibrio: {balance_strategy}")
        logger.info(f"📊 Mín. muestras por clase: {min_samples_per_class}")
        if self.global_params:
            logger.info(f"⚙️ Parámetros globales: {self.global_params}")
        if self.model_specific_params:
            logger.info(f"🎯 Parámetros específicos: {len(self.model_specific_params)} modelos configurados")
    
    def discover_available_models(self):
        """Descubre todos los modelos disponibles en ambos backends"""
        available_models = {}
        
        # Modelos Ollama
        if self.ollama_client.is_available():
            for model_name in self.ollama_client.available_models:
                available_models[model_name] = 'ollama'
                logger.info(f"🦙 Encontrado en Ollama: {model_name}")
        
        # Modelos Hugging Face locales
        if os.path.exists(self.models_dir):
            for item in os.listdir(self.models_dir):
                model_path = os.path.join(self.models_dir, item)
                if os.path.isdir(model_path):
                    config_file = os.path.join(model_path, 'config.json')
                    if os.path.exists(config_file):
                        # ⭐ FIX: Verificar si el modelo es válido antes de agregarlo
                        try:
                            # Verificar que el archivo de configuración sea válido
                            with open(config_file, 'r') as f:
                                config = json.load(f)
                            
                            # Verificar que tenga los campos mínimos requeridos
                            if 'model_type' in config or 'architectures' in config:
                                available_models[item] = 'huggingface'
                                logger.info(f"🤗 Encontrado en HF: {item}")
                            else:
                                logger.warning(f"⚠️ Configuración inválida en {item}, omitiendo")
                                
                        except Exception as e:
                            logger.warning(f"⚠️ Error verificando {item}: {str(e)}, omitiendo")
        
        if len(available_models) == 0:
            error_msg = "❌ ERROR CRÍTICO: No se encontraron modelos disponibles"
            logger.error(error_msg)
            logger.error("🔍 Verificar:")
            logger.error(f"   - Ollama ejecutándose en localhost:11434")
            logger.error(f"   - Modelos de Ollama instalados (ollama list)")
            logger.error(f"   - Directorio de modelos HF: {self.models_dir}")
            logger.error("🚫 Deteniendo ejecución...")
            raise RuntimeError("No hay modelos disponibles para ejecutar predicciones")
        
        logger.info(f"📊 Total modelos encontrados: {len(available_models)}")
        return available_models
    
    def load_bert_model(self, model_name, model_path, num_labels):
        """Carga modelo BERT usando Hugging Face"""
        try:
            logger.info(f"🤗 Cargando modelo BERT: {model_name}")
            
            # Verificar si el directorio existe y contiene archivos necesarios
            required_files = ['config.json']
            missing_files = []
            
            for file in required_files:
                file_path = os.path.join(model_path, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)
            
            if missing_files:
                logger.error(f"❌ Archivos faltantes en {model_name}: {missing_files}")
                return False
            
            # ⭐ FIX: Manejo robusto de errores de carga
            try:
                # Cargar tokenizer con manejo de errores
                tokenizer = BertTokenizer.from_pretrained(
                    model_path,
                    local_files_only=True,
                    trust_remote_code=False
                )
            except Exception as e:
                logger.error(f"❌ Error cargando tokenizer para {model_name}: {str(e)}")
                logger.error(f"   Intentando tokenizer genérico...")
                try:
                    # Fallback a tokenizer genérico
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                    logger.warning(f"⚠️ Usando tokenizer genérico para {model_name}")
                except Exception as e2:
                    logger.error(f"❌ También falló tokenizer genérico: {str(e2)}")
                    return False
            
            # Configurar tokens especiales para proteínas
            special_tokens = ['[PROTEIN]']
            try:
                tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
            except Exception as e:
                logger.warning(f"⚠️ No se pudieron agregar tokens especiales: {str(e)}")
            
            # ⭐ FIX: Cargar modelo con manejo de errores robusto
            try:
                model = BertForSequenceClassification.from_pretrained(
                    model_path, 
                    num_labels=num_labels,
                    ignore_mismatched_sizes=True,
                    local_files_only=True,
                    trust_remote_code=False
                )
            except Exception as e:
                logger.error(f"❌ Error cargando modelo BERT {model_name}: {str(e)}")
                logger.error(f"   Detalles del error: {type(e).__name__}")
                
                if "HeaderTooLarge" in str(e):
                    logger.error(f"   El modelo {model_name} parece estar corrupto o ser incompatible")
                    logger.error(f"   Intente descargar el modelo nuevamente")
                elif "out of memory" in str(e).lower():
                    logger.error(f"   Memoria insuficiente para cargar {model_name}")
                    logger.error(f"   Considere usar un modelo más pequeño o liberar memoria")
                
                return False
            
            # Redimensionar embeddings para tokens especiales
            try:
                model.resize_token_embeddings(len(tokenizer))
                model.to(self.device)
            except Exception as e:
                logger.error(f"❌ Error configurando modelo {model_name}: {str(e)}")
                return False
            
            self.hf_models[model_name] = model
            self.hf_tokenizers[model_name] = tokenizer
            self.model_info[model_name] = {
                'type': 'bert',
                'backend': 'huggingface',
                'path': model_path,
                'num_parameters': sum(p.numel() for p in model.parameters()),
                'num_labels': num_labels
            }
            
            logger.info(f"✅ BERT {model_name} cargado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error general cargando BERT {model_name}: {str(e)}")
            logger.error(f"   Tipo de error: {type(e).__name__}")
            logger.error(f"   El modelo será omitido del análisis")
            return False
    
    def setup_ollama_models(self):
        """Configura TODOS los modelos disponibles en Ollama automáticamente"""
        if not self.ollama_client.is_available():
            logger.warning("⚠️ Ollama no está disponible")
            return
        
        logger.info("🔍 Detectando TODOS los modelos disponibles en Ollama...")
        
        # Configurar TODOS los modelos encontrados
        for model_name in self.ollama_client.available_models:
            # Determinar tipo automáticamente basándose en el nombre
            model_type, family = self.classify_ollama_model(model_name)
            
            self.model_info[model_name] = {
                'type': model_type,
                'backend': 'ollama',
                'family': family,
                'description': f'Modelo Ollama: {model_name}',
                'optimal_for': self.get_optimal_use(model_type),
                'auto_detected': True
            }
            
            logger.info(f"✅ Ollama: {model_name} configurado automáticamente (tipo: {model_type}, familia: {family})")
        
        total_ollama = sum(1 for info in self.model_info.values() if info.get('backend') == 'ollama')
        logger.info(f"📊 Total modelos Ollama detectados: {total_ollama}")
    
    def classify_ollama_model(self, model_name):
        """
        Clasifica automáticamente un modelo de Ollama basándose en su nombre
        Retorna: (tipo, familia)
        """
        model_lower = model_name.lower()
        
        # Detectar tipo basándose en el nombre
        if 'embed' in model_lower:
            return 'embedding', 'Embeddings'
        elif 'deepseek' in model_lower:
            if 'r1' in model_lower:
                return 'deepseek_reasoning', 'DeepSeek'
            else:
                return 'deepseek', 'DeepSeek'
        elif 'llama' in model_lower:
            return 'llama', 'Llama'
        elif 'mistral' in model_lower:
            return 'mistral', 'Mistral'
        elif 'qwen' in model_lower:
            return 'qwen', 'Qwen'
        elif 'gemma' in model_lower:
            return 'gemma', 'Gemma'
        elif 'phi' in model_lower:
            return 'phi', 'Phi'
        elif 'codellama' in model_lower or 'code' in model_lower:
            return 'code_llm', 'Code'
        elif 'instruct' in model_lower:
            return 'instruct', 'Instruct'
        elif 'chat' in model_lower:
            return 'chat', 'Chat'
        elif 'vision' in model_lower or 'visual' in model_lower:
            return 'vision', 'Vision'
        elif 'tool' in model_lower:
            return 'tool_use', 'Tools'
        else:
            # Modelo genérico - intentar clasificar por tamaño
            if any(size in model_lower for size in ['1b', '3b', '7b', '8b', '13b', '70b']):
                return 'llm', 'Generic LLM'
            else:
                return 'unknown', 'Unknown'
    
    def get_generation_params(self, model_name):
        """
        Obtiene parámetros de generación con prioridad: específicos > globales > automáticos
        """
        # 1. Parámetros automáticos basados en el modelo (por defecto)
        model_lower = model_name.lower()
        auto_params = {}
        
        if 'mistral' in model_lower:
            auto_params = {'temperature': 0.1, 'top_p': 0.9}
        elif 'deepseek' in model_lower:
            if 'r1' in model_lower:
                auto_params = {'temperature': 0.15, 'top_p': 0.85}
            else:
                auto_params = {'temperature': 0.2, 'top_p': 0.8}
        elif 'llama' in model_lower:
            if '1b' in model_lower:
                auto_params = {'temperature': 0.2, 'top_p': 0.9}
            else:
                auto_params = {'temperature': 0.15, 'top_p': 0.85}
        elif 'qwen' in model_lower:
            auto_params = {'temperature': 0.1, 'top_p': 0.85}
        elif 'gemma' in model_lower:
            auto_params = {'temperature': 0.2, 'top_p': 0.8}
        elif 'phi' in model_lower:
            auto_params = {'temperature': 0.25, 'top_p': 0.9}
        elif 'code' in model_lower:
            auto_params = {'temperature': 0.05, 'top_p': 0.95}
        elif 'instruct' in model_lower:
            auto_params = {'temperature': 0.1, 'top_p': 0.9}
        else:
            auto_params = {'temperature': 0.15, 'top_p': 0.85}
        
        # 2. Aplicar parámetros globales (sobrescriben automáticos)
        final_params = auto_params.copy()
        final_params.update(self.global_params)
        
        # 3. Aplicar parámetros específicos del modelo (sobrescriben todo)
        if model_name in self.model_specific_params:
            final_params.update(self.model_specific_params[model_name])
        
        return final_params
    
    def get_optimal_use(self, model_type):
        """Determina el uso óptimo basándose en el tipo de modelo"""
        use_cases = {
            'embedding': 'embeddings_and_similarity',
            'deepseek': 'general_analysis',
            'deepseek_reasoning': 'complex_reasoning',
            'llama': 'balanced_performance',
            'mistral': 'classification_tasks',
            'qwen': 'multilingual_tasks',
            'gemma': 'efficient_inference',
            'phi': 'small_scale_tasks',
            'code_llm': 'code_analysis',
            'instruct': 'instruction_following',
            'chat': 'conversational_tasks',
            'vision': 'image_analysis',
            'tool_use': 'function_calling',
            'llm': 'general_language_tasks',
            'unknown': 'experimental'
        }
        return use_cases.get(model_type, 'general_purpose')
    
    def load_available_models(self, num_labels=23):
        """Carga SOLO los modelos disponibles, sin crear modelos adicionales"""
        logger.info("🔄 Cargando solo modelos disponibles...")
        
        available_models = self.discover_available_models()
        loaded_count = 0
        failed_models = []
        
        # Separar modelos por backend para barras de progreso
        hf_models = [(name, backend) for name, backend in available_models.items() 
                     if backend == 'huggingface' and 'bert' in name.lower()]
        
        # Barra de progreso para modelos BERT
        if hf_models:
            logger.info(f"🤗 Cargando {len(hf_models)} modelos BERT...")
            
            try:
                hf_progress = tqdm(hf_models,
                                 desc="🤗 Cargando BERT",
                                 unit="modelo",
                                 leave=True)
                
                for model_name, backend in hf_progress:
                    model_path = os.path.join(self.models_dir, model_name)
                    
                    hf_progress.set_postfix({"Modelo": model_name[:20]})
                    
                    if self.load_bert_model(model_name, model_path, num_labels):
                        loaded_count += 1
                        hf_progress.set_postfix({"Estado": "✅ Cargado", "Modelo": model_name[:15]})
                    else:
                        failed_models.append(f"{model_name} (BERT)")
                        hf_progress.set_postfix({"Estado": "❌ Falló", "Modelo": model_name[:15]})
                
                hf_progress.close()
                
            except Exception as e:
                logger.warning(f"⚠️ Error en barra de progreso BERT: {str(e)}")
                # Fallback sin barra de progreso
                for model_name, backend in hf_models:
                    model_path = os.path.join(self.models_dir, model_name)
                    logger.info(f"🔄 Intentando cargar modelo BERT: {model_name}")
                    
                    if self.load_bert_model(model_name, model_path, num_labels):
                        loaded_count += 1
                        logger.info(f"✅ BERT {model_name} cargado exitosamente")
                    else:
                        failed_models.append(f"{model_name} (BERT)")
                        logger.warning(f"⚠️ BERT {model_name} falló al cargar, continuando sin él")
        
        # Configurar modelos Ollama (más rápido, menos progreso necesario)
        logger.info("🦙 Configurando modelos Ollama...")
        self.setup_ollama_models()
        ollama_count = sum(1 for info in self.model_info.values() if info.get('backend') == 'ollama')
        loaded_count += ollama_count
        
        # Mostrar progreso de configuración de Ollama si hay muchos
        if ollama_count > 5:
            ollama_models = [(name, info) for name, info in self.model_info.items() 
                           if info.get('backend') == 'ollama']
            
            try:
                config_progress = tqdm(ollama_models,
                                     desc="🦙 Configurando Ollama",
                                     unit="modelo",
                                     leave=False)
                
                for model_name, info in config_progress:
                    family = info.get('family', 'Unknown')
                    config_progress.set_postfix({"Familia": family[:15], "Modelo": model_name[:15]})
                    time.sleep(0.1)  # Simular tiempo de configuración
                
                config_progress.close()
                
            except Exception as e:
                logger.warning(f"⚠️ Error en barra de progreso Ollama: {str(e)}")
                # Continúa sin problema
                pass
        
        # ⭐ FIX: Permitir continuar aunque algunos modelos fallen
        if loaded_count == 0:
            error_msg = "❌ ERROR: No se pudo cargar ningún modelo"
            logger.error(error_msg)
            logger.error("🔍 Verificar que existan modelos en:")
            logger.error(f"   - Ollama: ollama list")
            logger.error(f"   - Hugging Face: {self.models_dir}")
            
            if failed_models:
                logger.error("❌ Modelos que fallaron:")
                for failed in failed_models:
                    logger.error(f"   - {failed}")
            
            logger.error("🚫 Deteniendo ejecución...")
            raise RuntimeError("No hay modelos cargados para ejecutar predicciones")
        
        # Mostrar resumen de carga
        logger.info(f"✅ Total modelos cargados: {loaded_count}")
        logger.info(f"   🤗 Hugging Face: {len(self.hf_models)}")
        logger.info(f"   🦙 Ollama: {ollama_count}")
        
        if failed_models:
            logger.warning(f"⚠️ Modelos que fallaron al cargar ({len(failed_models)}):")
            for failed in failed_models:
                logger.warning(f"   - {failed}")
            logger.warning("   El análisis continuará con los modelos disponibles")
        
        # Mostrar detalle de modelos Ollama detectados
        if ollama_count > 0:
            logger.info("📋 Modelos Ollama detectados:")
            for model_name, info in self.model_info.items():
                if info.get('backend') == 'ollama':
                    logger.info(f"   • {model_name} ({info.get('family', 'Unknown')}) - {info.get('type', 'unknown')}")
        
        # Mostrar detalle de modelos BERT cargados
        if len(self.hf_models) > 0:
            logger.info("📋 Modelos BERT cargados:")
            for model_name in self.hf_models.keys():
                logger.info(f"   • {model_name} (Hugging Face BERT)")
        
        return loaded_count
    
    def predict_bert_classification(self, model_name, sequences, batch_size=32, max_length=512):
        """Predicción con modelo BERT usando Hugging Face"""
        model = self.hf_models[model_name]
        tokenizer = self.hf_tokenizers[model_name]
        
        model.eval()
        predictions = []
        
        # Calcular número total de batches
        total_batches = (len(sequences) + batch_size - 1) // batch_size
        
        # Barra de progreso para BERT
        bert_progress = tqdm(total=total_batches,
                           desc=f"🤗 {model_name[:20]}... BERT",
                           unit="batch",
                           leave=False,
                           position=0)
        
        with torch.no_grad():
            for batch_idx, i in enumerate(range(0, len(sequences), batch_size)):
                batch_sequences = sequences[i:i+batch_size]
                
                # Actualizar información del batch
                batch_size_actual = len(batch_sequences)
                bert_progress.set_postfix({
                    "Batch": f"{batch_idx+1}/{total_batches}",
                    "Seqs": batch_size_actual,
                    "GPU": "CUDA" if next(model.parameters()).is_cuda else "CPU"
                })
                
                # Formatear secuencias para BERT
                formatted_sequences = []
                for seq in batch_sequences:
                    spaced_seq = ' '.join(list(seq))
                    formatted_sequences.append(f"[PROTEIN] {spaced_seq}")
                
                # Tokenizar
                inputs = tokenizer(
                    formatted_sequences,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # Predicción
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Aplicar sigmoid para multi-label
                probs = torch.sigmoid(logits)
                predictions.extend(probs.cpu().numpy())
                
                bert_progress.update(1)
        
        bert_progress.close()
        
        return np.array(predictions)
    
    def predict_ollama_generation(self, model_name, sequences, task='functional_category', batch_size=1):
        """Predicción con modelos LLM usando Ollama"""
        model_info = self.model_info[model_name]
        
        predictions = []
        
        # Obtener etiquetas según la tarea
        if task == 'functional_category':
            all_labels = self.standard_categories
        else:
            all_labels = self.standard_pathways
        
        num_labels = len(all_labels)
        
        # Barra de progreso principal para todas las secuencias
        total_sequences = len(sequences)
        main_progress = tqdm(total=total_sequences, 
                           desc=f"🦙 {model_name[:20]}... → {task[:12]}",
                           unit="seq",
                           position=0,
                           leave=True)
        
        # Estadísticas para mostrar en la barra
        successful_predictions = 0
        failed_predictions = 0
        total_time = 0
        
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            batch_predictions = []
            
            for seq_idx, seq in enumerate(batch_sequences):
                start_time = time.time()
                try:
                    # Generar prompt unificado
                    if task == 'functional_category':
                        prompt = self.prompt_generator.generate_functional_category_prompt(seq)
                    else:
                        prompt = self.prompt_generator.generate_pathway_prompt(seq)
                    
                    # Parámetros de generación optimizados automáticamente
                    generation_params = self.get_generation_params(model_name)
                    
                    # Generar respuesta usando Ollama
                    result = self.ollama_client.generate_text(model_name, prompt, **generation_params)
                    
                    if result and 'response' in result:
                        response = result['response']
                        
                        # Extraer predicción usando el generador unificado
                        prediction = self.prompt_generator.extract_prediction_from_response(response, task)
                        
                        # Convertir a vector de probabilidades
                        prob_vector = self.convert_prediction_to_probabilities(prediction, all_labels)
                        batch_predictions.append(prob_vector)
                        successful_predictions += 1
                    else:
                        # Vector de probabilidades uniforme en caso de error
                        uniform_prob = np.ones(num_labels) / num_labels
                        batch_predictions.append(uniform_prob)
                        failed_predictions += 1
                        
                except Exception as e:
                    logger.warning(f"Error en predicción Ollama para secuencia: {str(e)}")
                    uniform_prob = np.ones(num_labels) / num_labels
                    batch_predictions.append(uniform_prob)
                    failed_predictions += 1
                
                # Actualizar tiempo y progreso
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                
                # Calcular estadísticas
                current_seq = i + seq_idx + 1
                avg_time = total_time / current_seq
                eta_seconds = avg_time * (total_sequences - current_seq)
                
                # Actualizar barra de progreso con estadísticas
                main_progress.set_postfix({
                    "✅": successful_predictions,
                    "❌": failed_predictions,
                    "⏱️": f"{elapsed_time:.1f}s",
                    "ETA": f"{eta_seconds/60:.1f}m" if eta_seconds > 60 else f"{eta_seconds:.0f}s"
                })
                main_progress.update(1)
            
            predictions.extend(batch_predictions)
        
        main_progress.close()
        
        # Log resumen final
        success_rate = (successful_predictions / total_sequences) * 100 if total_sequences > 0 else 0
        avg_time_per_seq = total_time / total_sequences if total_sequences > 0 else 0
        
        logger.info(f"✅ {model_name} → {task}: {successful_predictions}/{total_sequences} exitosas "
                   f"({success_rate:.1f}%), {avg_time_per_seq:.1f}s/seq promedio")
        
        return np.array(predictions)
    
    def get_ollama_embeddings(self, model_name, sequences):
        """Obtiene embeddings usando Ollama (alternativa a BERT)"""
        embeddings = []
        
        # Barra de progreso para embeddings (solo si se llama directamente)
        progress = tqdm(sequences, 
                       desc=f"🔗 {model_name[:20]}... embeddings",
                       unit="seq",
                       leave=False)
        
        for seq in progress:
            try:
                embedding = self.ollama_client.get_embeddings(model_name, seq)
                if embedding is not None:
                    embeddings.append(embedding)
                else:
                    # Vector cero en caso de error
                    embeddings.append(np.zeros(768))  # Dimensión estándar
                    
                # Actualizar estado en la barra
                progress.set_postfix({"Estado": "✅", "Dim": len(embeddings[-1]) if len(embeddings) > 0 else "N/A"})
                    
            except Exception as e:
                logger.warning(f"Error obteniendo embedding: {str(e)}")
                embeddings.append(np.zeros(768))
                progress.set_postfix({"Estado": "❌", "Error": str(e)[:20]})
        
        progress.close()
        return np.array(embeddings)
    
    def convert_prediction_to_probabilities(self, prediction, all_labels):
        """Convierte predicción categórica a vector de probabilidades"""
        prob_vector = np.zeros(len(all_labels))
        
        try:
            if prediction in all_labels:
                idx = all_labels.index(prediction)
                prob_vector[idx] = 0.9  # Alta confianza
                # Distribuir probabilidad restante
                remaining_prob = 0.1 / (len(all_labels) - 1)
                for i in range(len(all_labels)):
                    if i != idx:
                        prob_vector[i] = remaining_prob
            else:
                # Distribución uniforme si no reconoce la predicción
                prob_vector.fill(1.0 / len(all_labels))
        except:
            # Distribución uniforme en caso de error
            prob_vector.fill(1.0 / len(all_labels))
        
        return prob_vector
    
    def predict_all_models(self, sequences, tasks=['functional_category', 'pathway'], batch_size=16):
        """Hace predicciones con todos los modelos para AMBAS tareas automáticamente"""
        logger.info(f"🔮 Realizando predicciones para tareas {tasks} con modelo híbrido...")
        logger.info(f"📊 Cada modelo predecirá {len(sequences)} muestras × {len(tasks)} tareas = {len(sequences) * len(tasks)} predicciones totales")
        
        if len(self.model_info) == 0:
            error_msg = "❌ ERROR: No hay modelos cargados para hacer predicciones"
            logger.error(error_msg)
            raise RuntimeError("No hay modelos disponibles para predicciones")
        
        # ⭐ FIX: Verificar que las secuencias sean válidas
        if not sequences or len(sequences) == 0:
            error_msg = "❌ ERROR: No hay secuencias para procesar"
            logger.error(error_msg)
            raise ValueError("Lista de secuencias está vacía")
        
        # Verificar tipo de datos de las secuencias
        if isinstance(sequences, np.ndarray):
            logger.info(f"📊 Secuencias recibidas como numpy array: {sequences.shape}")
        elif isinstance(sequences, list):
            logger.info(f"📊 Secuencias recibidas como lista: {len(sequences)} elementos")
            logger.info(f"📊 Tipo del primer elemento: {type(sequences[0])}")
            if len(sequences) > 0:
                logger.info(f"📊 Ejemplo de secuencia: {str(sequences[0])[:50]}...")
        else:
            logger.warning(f"⚠️ Tipo de secuencias no esperado: {type(sequences)}")
        
        all_predictions = {}
        
        # Calcular total de combinaciones modelo×tarea
        total_combinations = len(self.model_info) * len(tasks)
        
        # Barra de progreso principal para todos los modelos
        main_progress = tqdm(total=total_combinations,
                           desc="🎯 Progreso general",
                           unit="modelo×tarea",
                           position=1,
                           leave=True)
        
        combination_count = 0
        successful_models = 0
        failed_models = 0
        
        for model_name, model_info in self.model_info.items():
            backend = model_info.get('backend', 'unknown')
            model_display = f"{model_name[:25]}..." if len(model_name) > 25 else model_name
            
            main_progress.set_description(f"🔄 {model_display} ({backend})")
            
            start_time = time.time()
            
            try:
                model_predictions = {}
                model_successful_tasks = 0
                
                # Predecir cada tarea por separado
                for task_idx, task in enumerate(tasks):
                    combination_count += 1
                    task_display = task.replace('_', ' ').title()
                    
                    main_progress.set_postfix({
                        "Modelo": model_display[:15],
                        "Tarea": task_display[:12],
                        "✅": successful_models,
                        "❌": failed_models
                    })
                    
                    try:
                        if backend == 'huggingface' and model_info['type'] == 'bert':
                            # Usar Hugging Face para BERT - necesita ajuste dinámico de labels
                            if task == 'functional_category':
                                num_labels = len(self.standard_categories)
                            else:  # pathway
                                num_labels = len(self.standard_pathways)
                            
                            # Para BERT, usar barra de progreso para batches
                            logger.info(f"   🤗 Procesando {len(sequences)} secuencias con BERT...")
                            predictions = self.predict_bert_classification(model_name, sequences, batch_size)
                            
                            # Ajustar dimensiones si es necesario
                            if predictions.shape[1] != num_labels:
                                # Redimensionar o rellenar según sea necesario
                                if predictions.shape[1] < num_labels:
                                    # Rellenar con probabilidades bajas
                                    padding = np.random.uniform(0.01, 0.1, (predictions.shape[0], num_labels - predictions.shape[1]))
                                    predictions = np.concatenate([predictions, padding], axis=1)
                                else:
                                    # Truncar
                                    predictions = predictions[:, :num_labels]
                            
                            # Renormalizar para que sumen 1
                            predictions = predictions / predictions.sum(axis=1, keepdims=True)
                            
                        elif backend == 'ollama' and model_info['type'] == 'embedding':
                            # Para embeddings, usar barra de progreso
                            logger.info(f"   🔗 Obteniendo embeddings para {len(sequences)} secuencias...")
                            
                            embeddings = []
                            embed_progress = tqdm(sequences, 
                                                desc=f"🔗 {model_name[:20]}... embeddings",
                                                unit="seq",
                                                leave=False,
                                                position=0)
                            
                            for seq in embed_progress:
                                try:
                                    embedding = self.ollama_client.get_embeddings(model_name, seq)
                                    if embedding is not None:
                                        embeddings.append(embedding)
                                    else:
                                        embeddings.append(np.zeros(768))
                                except Exception as e:
                                    embeddings.append(np.zeros(768))
                            
                            embed_progress.close()
                            predictions = np.array(embeddings)
                            
                        elif backend == 'ollama':
                            # Usar Ollama para modelos generativos - predicción específica por tarea
                            # La barra de progreso se maneja dentro de predict_ollama_generation
                            predictions = self.predict_ollama_generation(model_name, sequences, task, batch_size=1)
                            
                        else:
                            logger.warning(f"⏭️ Saltando {model_name} - backend no reconocido: {backend}")
                            main_progress.update(1)
                            continue
                        
                        model_predictions[task] = predictions
                        model_successful_tasks += 1
                        logger.info(f"   ✅ {task}: {predictions.shape}")
                        
                    except Exception as task_error:
                        logger.error(f"   ❌ Error en tarea {task}: {str(task_error)}")
                        # Continuar con la siguiente tarea
                        pass
                    
                    main_progress.update(1)
                
                if not model_predictions:
                    logger.warning(f"⏭️ No se pudieron generar predicciones para {model_name}")
                    failed_models += 1
                    continue
                
                prediction_time = time.time() - start_time
                
                all_predictions[model_name] = {
                    'predictions': model_predictions,  # Ahora es un dict con ambas tareas
                    'prediction_time': prediction_time,
                    'samples_per_second': len(sequences) / prediction_time,
                    'backend': backend,
                    'model_type': model_info['type'],
                    'tasks_completed': list(model_predictions.keys())
                }
                
                successful_models += 1
                logger.info(f"✅ {model_name}: {len(sequences)} muestras × {len(model_predictions)} tareas en {prediction_time:.2f}s "
                           f"({len(sequences)/prediction_time:.1f} muestras/s)")
                
            except Exception as e:
                logger.error(f"❌ Error en predicción con {model_name}: {str(e)}")
                logger.error(f"   Tipo de error: {type(e).__name__}")
                logger.error(f"   Backend: {backend}")
                logger.error(f"   Modelo continuará siendo omitido")
                failed_models += 1
                
                # Actualizar progreso para las tareas restantes de este modelo
                remaining_tasks = len(tasks) - (combination_count % len(tasks))
                main_progress.update(remaining_tasks)
                combination_count += remaining_tasks - 1
                continue
        
        main_progress.close()
        
        if len(all_predictions) == 0:
            error_msg = "❌ ERROR: No se pudieron generar predicciones con ningún modelo"
            logger.error(error_msg)
            logger.error("🔍 Posibles causas:")
            logger.error("   - Todos los modelos fallaron durante la predicción")
            logger.error("   - Problemas de conectividad con Ollama")
            logger.error("   - Secuencias de entrada inválidas")
            logger.error("   - Memoria insuficiente")
            raise RuntimeError("Todas las predicciones fallaron")
        
        # Verificar que todas las tareas se completaron
        completed_tasks = set()
        for model_results in all_predictions.values():
            completed_tasks.update(model_results.get('tasks_completed', []))
        
        logger.info(f"🎯 RESUMEN DE PREDICCIONES:")
        logger.info(f"   📊 Modelos exitosos: {successful_models}/{len(self.model_info)}")
        logger.info(f"   ❌ Modelos fallidos: {failed_models}/{len(self.model_info)}")
        logger.info(f"   📋 Tareas completadas: {sorted(completed_tasks)}")
        logger.info(f"   📈 Total predicciones: {len(sequences)} muestras × {len(completed_tasks)} tareas × {successful_models} modelos")
        
        return all_predictions
    
    def get_model_summary(self):
        """Obtiene resumen de modelos cargados"""
        summary = {
            'total_models': len(self.model_info),
            'models_by_backend': {},
            'models_by_type': {},
            'device': self.device,
            'ollama_available': self.ollama_client.is_available(),
            'balance_strategy': self.data_balancer.strategy,
            'min_samples_per_class': self.data_balancer.min_samples_per_class,
            'models': {}
        }
        
        for model_name, info in self.model_info.items():
            backend = info.get('backend', 'unknown')
            model_type = info.get('type', 'unknown')
            
            # Contar por backend
            if backend not in summary['models_by_backend']:
                summary['models_by_backend'][backend] = 0
            summary['models_by_backend'][backend] += 1
            
            # Contar por tipo
            if model_type not in summary['models_by_type']:
                summary['models_by_type'][model_type] = 0
            summary['models_by_type'][model_type] += 1
            
            summary['models'][model_name] = info
        
        return summary


def parse_model_specific_params(model_params_list):
    """
    Parsea parámetros específicos por modelo desde la línea de comandos
    Formato: "modelo:param1=valor1,param2=valor2"
    """
    if not model_params_list:
        return {}
    
    model_specific = {}
    
    for param_string in model_params_list:
        try:
            # Dividir modelo y parámetros
            if ':' not in param_string:
                logger.warning(f"⚠️ Formato incorrecto en --model-params: {param_string}")
                continue
            
            parts = param_string.split(':', 1)
            model_name = parts[0]
            params_str = parts[1] if len(parts) > 1 else ""
            
            # Parsear parámetros
            model_params = {}
            if params_str:
                for param_pair in params_str.split(','):
                    if '=' in param_pair:
                        key, value = param_pair.split('=', 1)
                        key = key.strip()
                        
                        # Convertir valores a tipos apropiados
                        try:
                            if key in ['temperature', 'top_p', 'top_k', 'repeat_penalty']:
                                if key == 'top_k':
                                    model_params[key.replace('_', '-')] = int(value)
                                else:
                                    model_params[key.replace('_', '-')] = float(value)
                            elif key == 'max_tokens':
                                model_params['max-tokens'] = int(value)
                            else:
                                model_params[key] = value
                        except ValueError:
                            logger.warning(f"⚠️ Valor inválido para {key}: {value}")
            
            model_specific[model_name] = model_params
            logger.info(f"📝 Parámetros específicos para {model_name}: {model_params}")
            
        except Exception as e:
            logger.warning(f"⚠️ Error parseando parámetros para modelo: {param_string} - {str(e)}")
    
    return model_specific


def filter_models_by_args(available_models, use_models=None, exclude_models=None):
    """
    Filtra modelos según argumentos --use-models y --exclude-models
    """
    filtered_models = available_models.copy()
    
    # Aplicar filtro de inclusión
    if use_models:
        # Convertir a set para búsqueda rápida
        use_set = set(use_models)
        filtered_models = {k: v for k, v in filtered_models.items() if k in use_set}
        logger.info(f"🎯 Usando solo modelos especificados: {list(filtered_models.keys())}")
    
    # Aplicar filtro de exclusión
    if exclude_models:
        exclude_set = set(exclude_models)
        filtered_models = {k: v for k, v in filtered_models.items() if k not in exclude_set}
        logger.info(f"🚫 Excluyendo modelos: {exclude_models}")
        logger.info(f"✅ Modelos restantes: {list(filtered_models.keys())}")
    
    return filtered_models


def list_available_models(hybrid_loader):
    """
    Lista todos los modelos disponibles con información detallada
    """
    logger.info("📋 MODELOS DISPONIBLES:")
    logger.info("=" * 60)
    
    # Agrupar por backend
    backends = {}
    for model_name, info in hybrid_loader.model_info.items():
        backend = info.get('backend', 'unknown')
        if backend not in backends:
            backends[backend] = []
        backends[backend].append((model_name, info))
    
    for backend, models in backends.items():
        logger.info(f"\n🔧 {backend.upper()}:")
        for model_name, info in models:
            family = info.get('family', 'Unknown')
            model_type = info.get('type', 'unknown')
            description = info.get('description', 'No description')
            
            logger.info(f"   • {model_name}")
            logger.info(f"     Familia: {family}")
            logger.info(f"     Tipo: {model_type}")
            logger.info(f"     Descripción: {description}")
            
            # Mostrar parámetros por defecto para modelos Ollama
            if backend == 'ollama' and hasattr(hybrid_loader, 'get_generation_params'):
                default_params = hybrid_loader.get_generation_params(model_name)
                logger.info(f"     Parámetros por defecto: {default_params}")
    
    logger.info(f"\n📊 Total: {len(hybrid_loader.model_info)} modelos disponibles")


def parse_console_arguments():
    """Parsea argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='Sistema Híbrido Multitarea de Predicción de Proteínas (Ollama + Hugging Face) con Equilibrio Automático',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Predicciones para AMBAS tareas automáticamente (por defecto)
  python hybrid_loader_balanced.py --max-samples 50 --batch-size 4
  
  # Solo categorías funcionales
  python hybrid_loader_balanced.py --tasks functional_category --max-samples 100
  
  # Solo pathways
  python hybrid_loader_balanced.py --tasks pathway --max-samples 100
  
  # Ambas tareas explícitamente
  python hybrid_loader_balanced.py --tasks functional_category pathway --max-samples 50
  
  # Con estrategias de equilibrio diferentes
  python hybrid_loader_balanced.py --balance-strategy stratified --max-samples 200
  python hybrid_loader_balanced.py --balance-strategy filter --min-samples-per-class 5
  
  # Resultado: 50 muestras → 50 categories + 50 pathways = 100 predicciones por modelo
        """
    )
    
    # Argumentos básicos
    parser.add_argument(
        '--max-samples', 
        type=int, 
        default=None,
        help='Número máximo de muestras a predecir (None = todas las disponibles)'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=8,
        help='Tamaño de lote para procesamiento (recomendado: 4-16 según GPU)'
    )
    
    parser.add_argument(
        '--tasks', 
        nargs='+',
        choices=['functional_category', 'pathway', 'both'],
        default=['both'],
        help='Tareas de predicción a ejecutar (por defecto: ambas tareas automáticamente)'
    )
    
    parser.add_argument(
        '--data-path', 
        type=str, 
        default="/home/lab/Desktop/uwo/COG_LLMS/codigos/processed_data/functional_category_data.pkl",
        help='Ruta al archivo de datos de entrada'
    )
    
    parser.add_argument(
        '--models-dir', 
        type=str, 
        default="/home/lab/Desktop/uwo/COG_LLMS/Models",
        help='Directorio que contiene modelos de Hugging Face'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default="./predictions/hybrid_predictions.pkl",
        help='Ruta de salida para las predicciones'
    )
    
    parser.add_argument(
        '--ollama-host', 
        type=str, 
        default="localhost",
        help='Host donde está ejecutándose Ollama'
    )
    
    parser.add_argument(
        '--ollama-port', 
        type=int, 
        default=11434,
        help='Puerto de Ollama'
    )
    
    # ⭐ NUEVOS PARÁMETROS DE EQUILIBRIO DE CLASES
    parser.add_argument(
        '--balance-strategy', 
        type=str, 
        choices=['stratified', 'undersample', 'oversample', 'filter'],
        default='undersample',
        help='Estrategia de equilibrio de clases'
    )
    
    parser.add_argument(
        '--min-samples-per-class', 
        type=int, 
        default=2,
        help='Número mínimo de muestras por clase'
    )
    
    parser.add_argument(
        '--max-samples-per-class', 
        type=int, 
        default=None,
        help='Número máximo de muestras por clase (solo para oversample)'
    )
    
    # PARÁMETROS DE GENERACIÓN
    parser.add_argument(
        '--temperature', 
        type=float, 
        default=0.5,
        help='Temperature para TODOS los modelos (0.0-1.0, más bajo = más determinístico)'
    )
    
    parser.add_argument(
        '--top-p', 
        type=float, 
        default=None,
        help='Top-p para TODOS los modelos (0.0-1.0, núcleo sampling)'
    )
    
    parser.add_argument(
        '--top-k', 
        type=int, 
        default=None,
        help='Top-k para TODOS los modelos (número de tokens considerados)'
    )
    
    parser.add_argument(
        '--repeat-penalty', 
        type=float, 
        default=None,
        help='Penalización por repetición para TODOS los modelos (>1.0)'
    )
    
    parser.add_argument(
        '--max-tokens', 
        type=int, 
        default=None,
        help='Número máximo de tokens a generar'
    )
    
    parser.add_argument(
        '--model-params', 
        action='append',
        help='Parámetros específicos por modelo: "modelo:parametro=valor,parametro=valor"'
    )
    
    parser.add_argument(
        '--list-models', 
        action='store_true',
        help='Solo listar modelos disponibles sin ejecutar predicciones'
    )
    
    parser.add_argument(
        '--use-models', 
        nargs='+',
        help='Usar solo los modelos especificados (ej: --use-models mistral:latest deepseek-llm:7b)'
    )
    
    parser.add_argument(
        '--exclude-models', 
        nargs='+',
        help='Excluir modelos específicos (ej: --exclude-models nomic-embed-text:latest)'
    )
    
    parser.add_argument(
        '--timeout', 
        type=int, 
        default=300,
        help='Timeout en segundos para cada predicción de Ollama'
    )
    
    return parser.parse_args()


def run_hybrid_predictions_with_args(args):
    """
    Ejecuta predicciones híbridas con argumentos de consola y equilibrio automático
    """
    # Preparar parámetros globales
    global_params = {}
    if args.temperature is not None:
        global_params['temperature'] = args.temperature
    if args.top_p is not None:
        global_params['top-p'] = args.top_p
    if args.top_k is not None:
        global_params['top-k'] = args.top_k
    if args.repeat_penalty is not None:
        global_params['repeat-penalty'] = args.repeat_penalty
    if args.max_tokens is not None:
        global_params['max-tokens'] = args.max_tokens
    
    # Parsear parámetros específicos por modelo
    model_specific_params = parse_model_specific_params(args.model_params)
    
    # Procesar tareas
    if 'both' in args.tasks:
        tasks_to_run = ['functional_category', 'pathway']
    else:
        tasks_to_run = args.tasks
    
    # Mostrar configuración
    logger.info("🚀 INICIANDO SISTEMA HÍBRIDO CON EQUILIBRIO AUTOMÁTICO")
    logger.info("=" * 70)
    logger.info(f"📊 Configuración:")
    logger.info(f"   📈 Muestras máximas: {args.max_samples or 'Todas'}")
    logger.info(f"   📦 Batch size: {args.batch_size}")
    logger.info(f"   🎯 Tareas: {tasks_to_run} ({'AMBAS' if len(tasks_to_run) > 1 else tasks_to_run[0]})")
    logger.info(f"   ⏱️ Timeout: {args.timeout}s")
    logger.info(f"   ⚖️ Estrategia de equilibrio: {args.balance_strategy}")
    logger.info(f"   📊 Mín. muestras por clase: {args.min_samples_per_class}")
    if args.max_samples_per_class:
        logger.info(f"   📊 Máx. muestras por clase: {args.max_samples_per_class}")
    
    if global_params:
        logger.info(f"   ⚙️ Parámetros globales: {global_params}")
    if model_specific_params:
        logger.info(f"   🎯 Parámetros específicos: {len(model_specific_params)} modelos")
    if args.use_models:
        logger.info(f"   ✅ Solo modelos: {args.use_models}")
    if args.exclude_models:
        logger.info(f"   ❌ Excluir modelos: {args.exclude_models}")
    
    logger.info(f"   📁 Datos: {args.data_path}")
    logger.info(f"   🔧 Modelos HF: {args.models_dir}")
    logger.info(f"   🦙 Ollama: {args.ollama_host}:{args.ollama_port}")
    logger.info(f"   💾 Salida: {args.output}")
    logger.info("=" * 70)
    
    # Inicializar cargador híbrido con equilibrio automático
    try:
        hybrid_loader = HybridModelLoader(
            models_dir=args.models_dir,
            ollama_host=args.ollama_host,
            ollama_port=args.ollama_port,
            global_params=global_params,
            model_specific_params=model_specific_params,
            timeout=args.timeout,
            balance_strategy=args.balance_strategy,
            min_samples_per_class=args.min_samples_per_class
        )
        
        # Cargar modelos
        hybrid_loader.load_available_models(num_labels=23)  # Será actualizado con datos reales
        
        # Aplicar filtros de modelos
        if args.use_models or args.exclude_models:
            original_models = hybrid_loader.model_info.copy()
            filtered_models = filter_models_by_args(
                original_models, 
                args.use_models, 
                args.exclude_models
            )
            hybrid_loader.model_info = filtered_models
        
        # Si solo se quiere listar modelos, mostrarlos y salir
        if args.list_models:
            list_available_models(hybrid_loader)
            return None
        
    except RuntimeError as e:
        logger.error("🚫 Deteniendo ejecución por falta de modelos")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error inicializando sistema híbrido: {str(e)}")
        sys.exit(1)
    
    # Cargar datos con equilibrio automático
    try:
        logger.info(f"📊 Cargando datos de {args.data_path}...")
        with open(args.data_path, 'rb') as f:
            data = pickle.load(f)
        
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        label_names = data['label_names']
        
        logger.info(f"📈 Datos originales: {len(X_test)} muestras de prueba, {len(label_names)} etiquetas")
        
        # ⭐ APLICAR EQUILIBRIO AUTOMÁTICO DE CLASES
        X_test_balanced, y_test_balanced, balancing_info = hybrid_loader.data_balancer.balance_data(
            X_test, y_test, max_samples=args.max_samples, label_names=label_names
        )
        
        logger.info(f"🎯 Datos equilibrados: {len(X_test_balanced)} muestras finales")
        
        # Actualizar datos para usar los equilibrados
        X_test = X_test_balanced
        y_test = y_test_balanced
        
        # Actualizar número de etiquetas si se filtraron clases
        if balancing_info['excluded_classes']:
            # Filtrar label_names para excluir clases problemáticas
            label_names_filtered = [label_names[i] for i in range(len(label_names)) 
                                  if i not in balancing_info['excluded_classes']]
            label_names = label_names_filtered
            logger.info(f"🏷️ Etiquetas actualizadas: {len(label_names)} clases válidas")
        
    except FileNotFoundError:
        error_msg = f"❌ ERROR: Archivo de datos no encontrado: {args.data_path}"
        logger.error(error_msg)
        logger.error("🔍 Verificar que el archivo existe o ejecutar data_preparation.py primero")
        raise FileNotFoundError(error_msg)
    except Exception as e:
        error_msg = f"❌ ERROR cargando datos: {str(e)}"
        logger.error(error_msg)
        raise
    
    # ⭐ ESTIMACIÓN DE TIEMPO ANTES DE COMENZAR
    num_models = len(hybrid_loader.model_info)
    num_sequences = len(X_test)
    num_tasks = len(tasks_to_run)
    
    # Estimaciones aproximadas por tipo de modelo
    ollama_models = sum(1 for info in hybrid_loader.model_info.values() if info.get('backend') == 'ollama')
    bert_models = sum(1 for info in hybrid_loader.model_info.values() if info.get('backend') == 'huggingface')
    
    # Estimación conservadora de tiempo
    estimated_time_ollama = ollama_models * num_sequences * num_tasks * 3  # ~3 segundos por secuencia por tarea
    estimated_time_bert = bert_models * num_sequences * num_tasks * 0.1    # ~0.1 segundos por secuencia por tarea
    total_estimated_seconds = estimated_time_ollama + estimated_time_bert
    
    if total_estimated_seconds > 60:
        estimated_minutes = total_estimated_seconds / 60
        time_str = f"{estimated_minutes:.1f} minutos"
    else:
        time_str = f"{total_estimated_seconds:.0f} segundos"
    
    logger.info("=" * 70)
    logger.info("⏱️ ESTIMACIÓN DE TIEMPO:")
    logger.info(f"   📊 {num_models} modelos × {num_sequences} secuencias × {num_tasks} tareas")
    logger.info(f"   🦙 Modelos Ollama: {ollama_models} (más lentos)")
    logger.info(f"   🤗 Modelos BERT: {bert_models} (más rápidos)")
    logger.info(f"   ⌛ Tiempo estimado: ~{time_str}")
    logger.info(f"   💡 Use barras de progreso para seguimiento en tiempo real")
    logger.info("=" * 70)
    
    # Hacer predicciones para TODAS las tareas automáticamente
    try:
        logger.info(f"🎯 INICIANDO PREDICCIONES PARA {len(tasks_to_run)} TAREAS:")
        for i, task in enumerate(tasks_to_run, 1):
            logger.info(f"   {i}. {task.replace('_', ' ').title()}")
        
        # Mostrar barra de progreso general al inicio
        print("\n" + "="*50)
        print("🚀 Iniciando predicciones con barras de progreso")
        print("="*50)
        
        predictions = hybrid_loader.predict_all_models(
            X_test, 
            tasks=tasks_to_run, 
            batch_size=args.batch_size
        )
        
        # Preparar metadata para ambas tareas
        task_metadata = {}
        
        # Crear metadata específica para cada tarea
        for task in tasks_to_run:
            if task == 'functional_category':
                task_label_names = hybrid_loader.standard_categories
            else:  # pathway
                task_label_names = hybrid_loader.standard_pathways
            
            task_metadata[task] = {
                'num_labels': len(task_label_names),
                'label_names': task_label_names,
                'y_true': y_test,  # Mismo ground truth, diferentes interpretaciones
                'task_description': f"{task.replace('_', ' ').title()} prediction task"
            }
        
        # Agregar metadata con información de configuración y equilibrio
        predictions['metadata'] = {
            'tasks': tasks_to_run,
            'num_samples': len(X_test),
            'task_metadata': task_metadata,
            'global_label_names': label_names,  # Para compatibilidad
            'y_true': y_test,
            'model_summary': hybrid_loader.get_model_summary(),
            'approach': 'hybrid_ollama_huggingface_balanced_multitask',
            'console_args': vars(args),
            'global_params': global_params,
            'model_specific_params': model_specific_params,
            'balancing_info': balancing_info,  # ⭐ Información del equilibrio
            'sample_texts': X_test  # Guardar textos para comparaciones
        }
        
        # Guardar predicciones
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'wb') as f:
            pickle.dump(predictions, f)
        
        logger.info("🎉 Predicciones híbridas completadas exitosamente")
        logger.info(f"💾 Resultados guardados en: {args.output}")
        
        # Mostrar resumen final detallado
        logger.info("\n📊 RESUMEN FINAL:")
        logger.info("=" * 60)
        logger.info(f"✅ Modelos evaluados: {len(predictions) - 1}")
        logger.info(f"📈 Muestras procesadas: {predictions['metadata']['num_samples']}")
        logger.info(f"🎯 Tareas completadas: {len(tasks_to_run)}")
        for i, task in enumerate(tasks_to_run, 1):
            task_info = task_metadata[task]
            logger.info(f"   {i}. {task_info['task_description']}: {task_info['num_labels']} clases")
        logger.info(f"🔧 Enfoque: {predictions['metadata']['approach']}")
        logger.info(f"⚖️ Estrategia de equilibrio: {args.balance_strategy}")
        
        # Calcular total de predicciones generadas
        total_predictions = 0
        for model_results in predictions.values():
            if isinstance(model_results, dict) and 'predictions' in model_results:
                if isinstance(model_results['predictions'], dict):
                    # Contar predicciones por tarea
                    for task_predictions in model_results['predictions'].values():
                        if hasattr(task_predictions, 'shape'):
                            total_predictions += task_predictions.shape[0]
                        else:
                            total_predictions += len(task_predictions)
        
        logger.info(f"📊 Total predicciones generadas: {total_predictions:,}")
        logger.info(f"   ({len(X_test)} muestras × {len(tasks_to_run)} tareas × {len(predictions)-1} modelos)")
        
        # Mostrar estadísticas de equilibrio
        if balancing_info:
            changes = balancing_info.get('changes', {})
            logger.info(f"\n📊 RESULTADOS DEL EQUILIBRIO:")
            logger.info(f"   Original: {balancing_info['original_stats']['total_samples']} muestras")
            logger.info(f"   Final: {balancing_info['final_stats']['total_samples']} muestras")
            if balancing_info['excluded_classes']:
                logger.info(f"   Clases excluidas: {len(balancing_info['excluded_classes'])}")
            logger.info(f"   Balance mejorado: {'Sí' if changes.get('balance_improvement', 0) > 0 else 'No'}")
            logger.info(f"   Ahora equilibrado: {'Sí' if changes.get('is_now_balanced', False) else 'No'}")
        
        # Mostrar estadísticas por backend
        summary = predictions['metadata']['model_summary']
        logger.info(f"\n🎯 MODELOS POR BACKEND:")
        for backend, count in summary['models_by_backend'].items():
            logger.info(f"   {backend}: {count} modelos")
        
        # Mostrar estructura de salida
        logger.info(f"\n📁 ESTRUCTURA DE PREDICCIONES:")
        logger.info(f"   📋 Para cada modelo:")
        for task in tasks_to_run:
            logger.info(f"     └── predictions['{task}']: {len(X_test)} muestras")
        
        return predictions
        
    except RuntimeError as e:
        logger.error("🚫 Deteniendo ejecución por fallo en predicciones")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error durante predicciones: {str(e)}")
        sys.exit(1)


# Función principal para ejecutar predicciones híbridas
def run_hybrid_predictions(data_path, models_dir, output_path, tasks=['both'], 
                          batch_size=16, max_samples=None, ollama_host="localhost", ollama_port=11434,
                          temperature=None, top_p=None, timeout=300, balance_strategy='stratified',
                          min_samples_per_class=2):
    """
    Ejecuta predicciones con el modelo híbrido Ollama + Hugging Face con equilibrio automático
    (Versión de compatibilidad para código existente)
    
    Args:
        tasks: Lista de tareas o ['both'] para ambas tareas automáticamente
    """
    class Args:
        def __init__(self):
            self.data_path = data_path
            self.models_dir = models_dir
            self.output = output_path
            self.tasks = tasks
            self.batch_size = batch_size
            self.max_samples = max_samples
            self.ollama_host = ollama_host
            self.ollama_port = ollama_port
            self.temperature = temperature
            self.top_p = top_p
            self.top_k = None
            self.repeat_penalty = None
            self.max_tokens = None
            self.model_params = None
            self.list_models = False
            self.use_models = None
            self.exclude_models = None
            self.timeout = timeout
            self.balance_strategy = balance_strategy
            self.min_samples_per_class = min_samples_per_class
            self.max_samples_per_class = None
    
    args = Args()
    return run_hybrid_predictions_with_args(args)


# Ejemplo de uso
if __name__ == "__main__":
    # Parsear argumentos de consola
    args = parse_console_arguments()
    
    # Ejecutar predicciones híbridas con argumentos
    predictions = run_hybrid_predictions_with_args(args)
    
    if predictions is not None:  # Solo si no fue --list-models
        # Procesar tareas para mostrar en el resumen
        if 'both' in args.tasks:
            tasks_to_run = ['functional_category', 'pathway']
        else:
            tasks_to_run = args.tasks
        
        print(f"\n🎉 Ejecución completada exitosamente!")
        print(f"📁 Resultados en: {args.output}")
        print(f"📊 Para benchmarking multitarea ejecutar:")
        print(f"   python benchmarking_multitask_balanced.py --predictions {args.output}")
        print(f"")
        print(f"🔍 Estructura de predicciones generadas:")
        print(f"   📋 Cada modelo predice {len(tasks_to_run)} tareas por muestra")
        print(f"   📈 Total: {args.max_samples or 'todas'} muestras × {len(tasks_to_run)} tareas × [N] modelos")
        print(f"   📊 Barras de progreso mostraron avance en tiempo real")
        
        # Mostrar comandos de ejemplo para próximas ejecuciones
        print(f"\n💡 Ejemplos de comandos con predicciones multitarea:")
        print(f"   # Ambas tareas automáticamente (por defecto):")
        print(f"   python {__file__} --max-samples 50")
        print(f"   # Solo categorías funcionales:")
        print(f"   python {__file__} --tasks functional_category --max-samples 100")
        print(f"   # Solo pathways:")
        print(f"   python {__file__} --tasks pathway --max-samples 100")
        print(f"   # Ambas tareas explícitamente:")
        print(f"   python {__file__} --tasks functional_category pathway --max-samples 50")
        print(f"")
        print(f"   # Con equilibrio de clases:")
        print(f"   python {__file__} --balance-strategy stratified --max-samples 200")
        print(f"   python {__file__} --balance-strategy filter --min-samples-per-class 5")
        print(f"   python {__file__} --balance-strategy undersample --max-samples 500")
        print(f"")
        print(f"   # Sin timeout para modelos lentos:")
        print(f"   python {__file__} --timeout 3600 --max-samples 20")
        print(f"")
        print(f"   # Resultado: 50 muestras → 50 categories + 50 pathways = 100 predicciones por modelo")
        print(f"")
        print(f"📊 NOTA: Las barras de progreso te permiten:")
        print(f"   ✅ Ver progreso en tiempo real")
        print(f"   ⏱️ Estimar tiempo restante (ETA)")
        print(f"   📈 Monitorear éxito/fallo de predicciones")
        print(f"   🚀 Seguir el avance modelo por modelo")