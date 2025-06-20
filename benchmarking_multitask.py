import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
import os
import json
import argparse
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurar matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HybridModelBenchmarkMultitask:
    """
    Benchmarking especializado para el sistema híbrido MULTITAREA con equilibrio automático
    Soporta múltiples tareas: functional_category, pathway, etc.
    """
    def __init__(self, predictions_path):
        """
        Inicializa el benchmarking para modelos híbridos multitarea
        
        Args:
            predictions_path (str): Ruta al archivo de predicciones híbridas multitarea
        """
        self.predictions_path = predictions_path
        self.predictions = None
        self.metadata = None
        self.y_true = None
        self.label_names = None
        self.tasks = []
        self.task_metadata = {}
        self.results_by_task = {}  # Estructura: {task: {model: results}}
        self.sample_texts = None
        self.balancing_info = None
        
        self.load_predictions()
    
    def load_predictions(self):
        """Carga las predicciones guardadas del sistema híbrido multitarea"""
        logger.info(f"🔄 Cargando predicciones híbridas multitarea de {self.predictions_path}...")
        
        with open(self.predictions_path, 'rb') as f:
            data = pickle.load(f)
        
        self.predictions = {k: v for k, v in data.items() if k != 'metadata'}
        self.metadata = data['metadata']
        
        # Extraer información multitarea
        self.tasks = self.metadata.get('tasks', ['functional_category'])  # Default fallback
        self.task_metadata = self.metadata.get('task_metadata', {})
        self.balancing_info = self.metadata.get('balancing_info', None)
        
        # Ground truth y textos (globales para todas las tareas)
        self.y_true = self.metadata['y_true']
        self.label_names = self.metadata.get('global_label_names', self.metadata.get('label_names', []))
        self.sample_texts = self.metadata.get('sample_texts', None)
        
        if self.sample_texts is None:
            self.sample_texts = [f"Muestra_{i+1}" for i in range(len(self.y_true))]
        
        # Inicializar estructura de resultados por tarea
        for task in self.tasks:
            self.results_by_task[task] = {}
        
        logger.info(f"✅ Predicciones cargadas: {len(self.predictions)} modelos")
        logger.info(f"🎯 Tareas detectadas: {self.tasks}")
        logger.info(f"📊 Enfoque: {self.metadata.get('approach', 'unknown')}")
        logger.info(f"⚖️ Equilibrio aplicado: {'Sí' if self.balancing_info else 'No'}")
        
        # Mostrar información por tarea
        for task in self.tasks:
            task_info = self.task_metadata.get(task, {})
            task_labels = len(task_info.get('label_names', []))
            logger.info(f"   📋 {task}: {task_labels} clases")
    
    def identify_model_info(self, model_name):
        """
        Identifica información del modelo basado en su nombre y backend
        (Actualizado para el sistema híbrido multitarea)
        """
        model_data = self.predictions[model_name]
        backend = model_data.get('backend', 'unknown')
        model_type = model_data.get('model_type', 'unknown')
        
        # Determinar familia del modelo
        family = 'Unknown'
        display_name = model_name
        
        if backend == 'ollama':
            if 'deepseek' in model_name.lower():
                family = 'DeepSeek'
                if 'r1' in model_name:
                    display_name = 'DeepSeek R1 7B'
                else:
                    display_name = 'DeepSeek LLM 7B'
            elif 'llama' in model_name.lower():
                family = 'Llama'
                if '1b' in model_name:
                    display_name = 'Llama 3.2 1B'
                elif '3b' in model_name:
                    display_name = 'Llama 3.2 3B'
                else:
                    display_name = 'Llama 3.2'
            elif 'mistral' in model_name.lower():
                family = 'Mistral'
                display_name = 'Mistral 7B'
            elif 'qwen' in model_name.lower():
                family = 'Qwen'
                display_name = 'Qwen 2.5'
            elif 'gemma' in model_name.lower():
                family = 'Gemma'
                display_name = 'Gemma 2'
            elif 'phi' in model_name.lower():
                family = 'Phi'
                display_name = 'Phi 3.5'
            elif 'nomic' in model_name.lower():
                family = 'Embeddings'
                display_name = 'Nomic Embeddings'
        
        elif backend == 'huggingface':
            family = 'BERT'
            if 'distilbert' in model_name.lower():
                display_name = 'DistilBERT'
            elif 'bert-large' in model_name.lower():
                display_name = 'BERT Large'
            elif 'bert-base' in model_name.lower():
                display_name = 'BERT Base'
            else:
                display_name = model_name
        
        elif backend == 'sklearn':
            family = 'Baseline'
            if 'RandomForest' in model_name:
                display_name = 'Random Forest'
            elif 'LogisticRegression' in model_name:
                display_name = 'Logistic Regression'
            elif 'SVM' in model_name:
                display_name = 'Support Vector Machine'
            elif 'NaiveBayes' in model_name:
                display_name = 'Naive Bayes'
            else:
                display_name = model_name.replace('baseline_', '').replace('_', ' ')
        
        return {
            'family': family,
            'backend': backend,
            'model_type': model_type,
            'display_name': display_name,
            'original_name': model_name
        }
    
    def convert_y_true_to_single_label(self, y_true):
        """Convierte y_true multi-label a single-label"""
        if y_true.ndim == 2 and y_true.shape[1] > 1:
            return np.argmax(y_true, axis=1)
        else:
            return y_true
    
    def convert_to_single_label(self, y_prob, y_true=None):
        """Convierte predicciones multi-label a single-label"""
        if y_prob is None:
            return None, None, None
            
        # Convertir predicciones a single-label (argmax)
        if y_prob.ndim == 2 and y_prob.shape[1] > 1:
            y_pred_single = np.argmax(y_prob, axis=1)
            y_prob_single = y_prob
        else:
            if y_prob.ndim == 1:
                y_pred_single = (y_prob > 0.5).astype(int)
                y_prob_single = np.column_stack([1-y_prob, y_prob])
            else:
                y_pred_single = np.argmax(y_prob, axis=1)
                y_prob_single = y_prob
        
        # Convertir y_true si se proporciona
        y_true_single = None
        if y_true is not None:
            y_true_single = self.convert_y_true_to_single_label(y_true)
        
        return y_pred_single, y_prob_single, y_true_single
    
    def validate_and_align_dimensions(self, y_true, y_pred, y_prob=None, model_name="unknown", task="unknown"):
        """Valida y alinea las dimensiones entre ground truth y predicciones"""
        alignment_info = {
            'original_true_shape': y_true.shape if hasattr(y_true, 'shape') else len(y_true),
            'original_pred_shape': y_pred.shape if hasattr(y_pred, 'shape') else len(y_pred),
            'alignment_strategy': 'none',
            'samples_lost': 0,
            'has_issues': False
        }
        
        # Convertir a numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if y_prob is not None:
            y_prob = np.array(y_prob)
        
        n_true = len(y_true)
        n_pred = len(y_pred)
        
        logger.info(f"[{model_name}][{task}] Dimensiones - Ground truth: {y_true.shape}, Predicciones: {y_pred.shape}")
        
        if y_prob is not None:
            n_prob = len(y_prob)
            logger.info(f"[{model_name}][{task}] Probabilidades: {y_prob.shape}")
            
            if n_pred != n_prob:
                logger.warning(f"[{model_name}][{task}] ¡INCONSISTENCIA! Predicciones ({n_pred}) != Probabilidades ({n_prob})")
                alignment_info['has_issues'] = True
        
        # Alineamiento de dimensiones
        if n_true == n_pred:
            logger.info(f"[{model_name}][{task}] ✓ Dimensiones correctas: {n_true} muestras")
            return y_true, y_pred, y_prob, alignment_info
        
        elif n_true > n_pred:
            logger.warning(f"[{model_name}][{task}] ⚠️  PROBLEMA: Ground truth ({n_true}) > Predicciones ({n_pred})")
            alignment_info['has_issues'] = True
            alignment_info['samples_lost'] = n_true - n_pred
            alignment_info['alignment_strategy'] = 'truncate_true'
            
            y_true_aligned = y_true[:n_pred]
            y_pred_aligned = y_pred
            y_prob_aligned = y_prob
            
            logger.warning(f"[{model_name}][{task}] Truncando ground truth a {n_pred} muestras")
            
        else:  # n_pred > n_true
            logger.warning(f"[{model_name}][{task}] ⚠️  PROBLEMA: Predicciones ({n_pred}) > Ground truth ({n_true})")
            alignment_info['has_issues'] = True
            alignment_info['samples_lost'] = n_pred - n_true
            alignment_info['alignment_strategy'] = 'truncate_pred'
            
            y_true_aligned = y_true
            y_pred_aligned = y_pred[:n_true]
            y_prob_aligned = y_prob[:n_true] if y_prob is not None else None
            
            logger.warning(f"[{model_name}][{task}] Truncando predicciones a {n_true} muestras")
        
        final_n = len(y_true_aligned)
        logger.info(f"[{model_name}][{task}] ✓ Después del alineamiento: {final_n} muestras")
        
        return y_true_aligned, y_pred_aligned, y_prob_aligned, alignment_info
    
    def calculate_single_label_metrics_safe(self, y_true, y_pred, y_prob=None, model_name="unknown", task="unknown"):
        """Calcula métricas single-label con validación de dimensiones para una tarea específica"""
        
        # Validar y alinear dimensiones
        y_true_aligned, y_pred_aligned, y_prob_aligned, alignment_info = self.validate_and_align_dimensions(
            y_true, y_pred, y_prob, model_name, task
        )
        
        metrics = {}
        
        try:
            # Métricas básicas
            metrics['accuracy'] = accuracy_score(y_true_aligned, y_pred_aligned)
            
            # Precision, Recall, F1 (weighted average)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true_aligned, y_pred_aligned, average='weighted', zero_division=0
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1
            
            # Métricas macro
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                y_true_aligned, y_pred_aligned, average='macro', zero_division=0
            )
            metrics['precision_macro'] = precision_macro
            metrics['recall_macro'] = recall_macro
            metrics['f1_macro'] = f1_macro
            
            # Métricas por clase
            precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                y_true_aligned, y_pred_aligned, average=None, zero_division=0
            )
            metrics['precision_per_class'] = precision_per_class
            metrics['recall_per_class'] = recall_per_class
            metrics['f1_per_class'] = f1_per_class
            
            # AUC si hay probabilidades
            if y_prob_aligned is not None:
                try:
                    num_classes = len(np.unique(y_true_aligned))
                    
                    if num_classes == 2:
                        if y_prob_aligned.shape[1] == 2:
                            metrics['auc'] = roc_auc_score(y_true_aligned, y_prob_aligned[:, 1])
                        else:
                            metrics['auc'] = roc_auc_score(y_true_aligned, y_prob_aligned)
                    elif num_classes > 2:
                        y_true_bin = label_binarize(y_true_aligned, classes=range(num_classes))
                        if y_true_bin.shape[1] == 1:
                            metrics['auc'] = roc_auc_score(y_true_aligned, y_prob_aligned[:, 1])
                        else:
                            metrics['auc'] = roc_auc_score(y_true_bin, y_prob_aligned, average='weighted', multi_class='ovr')
                    else:
                        metrics['auc'] = np.nan
                        
                except Exception as e:
                    logger.warning(f"[{model_name}][{task}] No se pudo calcular AUC: {str(e)}")
                    metrics['auc'] = np.nan
            else:
                metrics['auc'] = np.nan
            
            metrics['alignment_info'] = alignment_info
            
            if alignment_info['has_issues']:
                logger.warning(f"[{model_name}][{task}] ⚠️  Métricas calculadas con datos alineados")
            else:
                logger.info(f"[{model_name}][{task}] ✓ Métricas calculadas correctamente")
                
        except Exception as e:
            logger.error(f"[{model_name}][{task}] ❌ Error calculando métricas: {str(e)}")
            metrics = {
                'accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan,
                'precision_macro': np.nan, 'recall_macro': np.nan, 'f1_macro': np.nan,
                'precision_per_class': np.array([]), 'recall_per_class': np.array([]),
                'f1_per_class': np.array([]), 'auc': np.nan,
                'alignment_info': alignment_info, 'error': str(e)
            }
        
        return metrics
    
    def extract_task_predictions(self, model_name, task):
        """
        Extrae predicciones para una tarea específica de un modelo
        
        Args:
            model_name: Nombre del modelo
            task: Nombre de la tarea ('functional_category', 'pathway', etc.)
            
        Returns:
            tuple: (y_prob, task_label_names) o (None, None) si no hay datos
        """
        model_data = self.predictions[model_name]
        
        # Estructura nueva: model_data['predictions'][task]
        if isinstance(model_data, dict) and 'predictions' in model_data:
            predictions_dict = model_data['predictions']
            
            # Verificar si es estructura multitarea
            if isinstance(predictions_dict, dict) and task in predictions_dict:
                y_prob = predictions_dict[task]
                task_info = self.task_metadata.get(task, {})
                task_label_names = task_info.get('label_names', self.label_names)
                return y_prob, task_label_names
            
            # Estructura legacy: solo una tarea
            elif not isinstance(predictions_dict, dict):
                if task == self.tasks[0]:  # Solo para la primera tarea
                    y_prob = predictions_dict
                    task_info = self.task_metadata.get(task, {})
                    task_label_names = task_info.get('label_names', self.label_names)
                    return y_prob, task_label_names
        
        # Estructura muy legacy: predicciones directas
        elif not isinstance(model_data, dict):
            if task == self.tasks[0]:  # Solo para la primera tarea
                y_prob = model_data
                task_info = self.task_metadata.get(task, {})
                task_label_names = task_info.get('label_names', self.label_names)
                return y_prob, task_label_names
        
        return None, None
    
    def calculate_all_metrics_multitask(self):
        """Calcula métricas para todos los modelos y todas las tareas"""
        logger.info("🔄 Calculando métricas para sistema híbrido multitarea...")
        
        # Convertir y_true a single-label
        y_true_single = self.convert_y_true_to_single_label(self.y_true)
        
        logger.info(f"📊 Ground truth shape: {y_true_single.shape}")
        logger.info(f"📈 Número de muestras: {len(y_true_single)}")
        logger.info(f"🎯 Tareas a procesar: {self.tasks}")
        
        total_combinations = len(self.predictions) * len(self.tasks)
        processed = 0
        
        for task in self.tasks:
            logger.info(f"📋 Procesando tarea: {task}")
            task_info = self.task_metadata.get(task, {})
            task_label_names = task_info.get('label_names', self.label_names)
            
            for model_name, model_data in self.predictions.items():
                processed += 1
                logger.info(f"[{processed}/{total_combinations}] {model_name} → {task}")
                
                try:
                    # Extraer predicciones para esta tarea
                    y_prob, extracted_labels = self.extract_task_predictions(model_name, task)
                    
                    if y_prob is None:
                        logger.warning(f"[{model_name}][{task}] ❌ No hay predicciones para esta tarea")
                        continue
                    
                    # Convertir a array
                    if not isinstance(y_prob, np.ndarray):
                        y_prob = np.array(y_prob)
                    
                    logger.info(f"[{model_name}][{task}] Predicciones shape: {y_prob.shape}")
                    
                    # Convertir a single-label
                    y_pred_single, y_prob_single, _ = self.convert_to_single_label(y_prob)
                    
                    if y_pred_single is None:
                        logger.error(f"[{model_name}][{task}] ❌ Error convirtiendo predicciones")
                        continue
                    
                    # Calcular métricas
                    metrics = self.calculate_single_label_metrics_safe(
                        y_true_single, y_pred_single, y_prob_single, model_name, task
                    )
                    
                    # Agregar información del modelo y tarea
                    if isinstance(model_data, dict):
                        metrics['prediction_time'] = model_data.get('prediction_time', np.nan)
                        metrics['samples_per_second'] = model_data.get('samples_per_second', np.nan)
                        metrics['backend'] = model_data.get('backend', 'unknown')
                        metrics['model_type'] = model_data.get('model_type', 'unknown')
                        metrics['tasks_completed'] = model_data.get('tasks_completed', [])
                    
                    # Información del modelo
                    model_info = self.identify_model_info(model_name)
                    metrics.update(model_info)
                    
                    # Información de la tarea
                    metrics['task'] = task
                    metrics['task_label_names'] = extracted_labels or task_label_names
                    metrics['num_task_labels'] = len(metrics['task_label_names'])
                    
                    # Guardar resultados alineados
                    alignment_info = metrics.get('alignment_info', {})
                    if alignment_info.get('has_issues', False):
                        y_true_aligned, y_pred_aligned, y_prob_aligned, _ = self.validate_and_align_dimensions(
                            y_true_single, y_pred_single, y_prob_single, model_name, task
                        )
                        self.results_by_task[task][model_name] = {
                            'metrics': metrics,
                            'y_pred': y_pred_aligned,
                            'y_prob': y_prob_aligned
                        }
                    else:
                        self.results_by_task[task][model_name] = {
                            'metrics': metrics,
                            'y_pred': y_pred_single,
                            'y_prob': y_prob_single
                        }
                    
                    logger.info(f"[{model_name}][{task}] ✅ Métricas calculadas exitosamente")
                    
                except Exception as e:
                    logger.error(f"[{model_name}][{task}] ❌ Error: {str(e)}")
                    # Crear entrada de error
                    self.results_by_task[task][model_name] = {
                        'metrics': {
                            'accuracy': np.nan, 'f1': np.nan, 'precision': np.nan, 'recall': np.nan,
                            'backend': 'unknown', 'display_name': model_name, 'task': task,
                            'error': str(e)
                        },
                        'y_pred': None, 'y_prob': None
                    }
        
        logger.info("✅ Métricas multitarea calculadas exitosamente")
        
        # Crear reporte de problemas
        self.create_multitask_alignment_report()
    
    def create_multitask_alignment_report(self):
        """Crea reporte de problemas de alineamiento para multitareas"""
        alignment_issues = []
        
        for task in self.tasks:
            for model_name, results in self.results_by_task[task].items():
                if 'metrics' in results and 'alignment_info' in results['metrics']:
                    info = results['metrics']['alignment_info']
                    
                    if info['has_issues']:
                        alignment_issues.append({
                            'model': model_name,
                            'task': task,
                            'strategy': info['alignment_strategy'],
                            'samples_lost': info['samples_lost'],
                            'original_true': info['original_true_shape'],
                            'original_pred': info['original_pred_shape']
                        })
        
        if alignment_issues:
            logger.warning("=" * 60)
            logger.warning("REPORTE DE PROBLEMAS DE ALINEAMIENTO MULTITAREA")
            logger.warning("=" * 60)
            
            for issue in alignment_issues:
                logger.warning(f"Modelo: {issue['model']} | Tarea: {issue['task']}")
                logger.warning(f"  Ground truth: {issue['original_true']} | Predicciones: {issue['original_pred']}")
                logger.warning(f"  Muestras perdidas: {issue['samples_lost']} | Estrategia: {issue['strategy']}")
                logger.warning("")
        
        return alignment_issues
    
    def create_multitask_performance_summary(self):
        """Crea resumen de rendimiento multitarea"""
        logger.info("📊 Creando resumen de rendimiento multitarea...")
        
        summary_data = []
        
        for task in self.tasks:
            for model_name, results in self.results_by_task[task].items():
                metrics = results['metrics']
                
                row = {
                    'Task': task,
                    'Model': metrics.get('display_name', model_name),
                    'Original_Name': model_name,
                    'Backend': metrics.get('backend', 'unknown'),
                    'Family': metrics.get('family', 'unknown'),
                    'Model_Type': metrics.get('model_type', 'unknown'),
                    'Prediction_Time': metrics.get('prediction_time', np.nan),
                    'Samples_Per_Second': metrics.get('samples_per_second', np.nan),
                    'Num_Task_Labels': metrics.get('num_task_labels', 0),
                    'Tasks_Completed': len(metrics.get('tasks_completed', [])),
                    'Accuracy': metrics.get('accuracy', np.nan),
                    'F1': metrics.get('f1', np.nan),
                    'F1_Macro': metrics.get('f1_macro', np.nan),
                    'Precision': metrics.get('precision', np.nan),
                    'Precision_Macro': metrics.get('precision_macro', np.nan),
                    'Recall': metrics.get('recall', np.nan),
                    'Recall_Macro': metrics.get('recall_macro', np.nan),
                    'AUC': metrics.get('auc', np.nan)
                }
                
                summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Ordenar por Tarea y luego por F1
        summary_df = summary_df.sort_values(['Task', 'F1'], ascending=[True, False], na_position='last')
        
        return summary_df
    
    def plot_multitask_comparison(self, save_path=None):
        """Crea gráfico comparativo multitarea"""
        summary_df = self.create_multitask_performance_summary()
        
        # Colores por backend
        backend_colors = {
            'ollama': '#FF6B35',
            'huggingface': '#4285F4', 
            'sklearn': '#34A853'
        }
        
        # Colores por tarea
        task_colors = {
            'functional_category': '#E91E63',
            'pathway': '#9C27B0'
        }
        
        # Métricas principales
        main_metrics = ['F1', 'Precision', 'Recall', 'Accuracy']
        available_metrics = [m for m in main_metrics if m in summary_df.columns and summary_df[m].notna().any()]
        
        n_metrics = len(available_metrics)
        n_tasks = len(self.tasks)
        
        # Crear figura con subplots por tarea
        fig, axes = plt.subplots(n_tasks, n_metrics, figsize=(5*n_metrics, 6*n_tasks))
        
        if n_tasks == 1 and n_metrics == 1:
            axes = [[axes]]
        elif n_tasks == 1:
            axes = [axes]
        elif n_metrics == 1:
            axes = [[ax] for ax in axes]
        
        for task_idx, task in enumerate(self.tasks):
            task_data = summary_df[summary_df['Task'] == task]
            
            for metric_idx, metric in enumerate(available_metrics):
                ax = axes[task_idx][metric_idx]
                
                data = task_data[['Model', 'Backend', metric]].dropna()
                
                if len(data) == 0:
                    ax.text(0.5, 0.5, f'Sin datos\npara {task}', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{metric} - {task.replace("_", " ").title()}')
                    continue
                
                # Crear barras coloreadas por backend
                bars = ax.bar(range(len(data)), data[metric], 
                             color=[backend_colors.get(b, '#gray') for b in data['Backend']])
                
                ax.set_title(f'{metric} - {task.replace("_", " ").title()}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Modelos')
                ax.set_ylabel(metric)
                ax.set_xticks(range(len(data)))
                ax.set_xticklabels(data['Model'], rotation=45, ha='right')
                
                # Añadir valores en barras
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    if not np.isnan(height):
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Leyenda por backend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=backend.title()) 
                          for backend, color in backend_colors.items() 
                          if backend in summary_df['Backend'].values]
        
        if legend_elements:
            fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                      ncol=len(legend_elements), title='Backend')
        
        plt.suptitle('Comparación de Rendimiento Multitarea\n(Sistema Híbrido con Equilibrio Automático)', 
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico multitarea guardado en {save_path}")
        
        plt.show()
        return fig
    
    def plot_task_confusion_matrices(self, task, save_path=None, colormap='Blues'):
        """Crea matrices de confusión para una tarea específica"""
        from sklearn.metrics import confusion_matrix
        
        if task not in self.results_by_task:
            logger.warning(f"Tarea '{task}' no encontrada")
            return None
        
        results = self.results_by_task[task]
        y_true_single = self.convert_y_true_to_single_label(self.y_true)
        
        # Obtener etiquetas de la tarea
        task_info = self.task_metadata.get(task, {})
        task_label_names = task_info.get('label_names', self.label_names)
        
        n_models = len(results)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        plot_idx = 0
        for model_name, model_results in results.items():
            if model_results['y_pred'] is None:
                continue
                
            y_pred = model_results['y_pred']
            y_true_aligned = y_true_single[:len(y_pred)]
            
            # Calcular matriz de confusión
            cm = confusion_matrix(y_true_aligned, y_pred)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plotear
            im = axes[plot_idx].imshow(cm_normalized, interpolation='nearest', cmap=colormap)
            axes[plot_idx].set_title(f'{model_results["metrics"]["display_name"]}', 
                                   fontsize=12, fontweight='bold')
            
            # Etiquetas
            tick_marks = np.arange(len(task_label_names))
            axes[plot_idx].set_xticks(tick_marks)
            axes[plot_idx].set_yticks(tick_marks)
            axes[plot_idx].set_xticklabels(task_label_names, rotation=45, ha='right')
            axes[plot_idx].set_yticklabels(task_label_names)
            
            # Colorbar
            plt.colorbar(im, ax=axes[plot_idx], fraction=0.046, pad=0.04)
            
            plot_idx += 1
        
        # Ocultar axes vacíos
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Matrices de Confusión - {task.replace("_", " ").title()}', 
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Matrices de confusión para {task} guardadas en {save_path}")
        
        plt.show()
        return fig
    
    def create_sample_comparisons_multitask(self, task, model_name, y_true_aligned, y_pred_aligned, y_prob_aligned=None, num_samples=10):
        """Crea comparaciones muestra por muestra para una tarea específica"""
        comparisons = []
        
        # Obtener etiquetas de la tarea
        task_info = self.task_metadata.get(task, {})
        task_label_names = task_info.get('label_names', self.label_names)
        
        total_samples = len(y_true_aligned)
        samples_to_show = min(num_samples, total_samples)
        
        if total_samples <= num_samples:
            sample_indices = list(range(total_samples))
        else:
            first_half = samples_to_show // 2
            second_half = samples_to_show - first_half
            sample_indices = list(range(first_half))
            if total_samples > samples_to_show:
                sample_indices.extend(list(range(total_samples - second_half, total_samples)))
        
        for idx in sample_indices:
            # Texto de la muestra
            if idx < len(self.sample_texts):
                sample_text = str(self.sample_texts[idx])
                if len(sample_text) > 100:
                    sample_text = sample_text[:97] + "..."
            else:
                sample_text = f"Muestra_{idx+1}"
            
            # Etiquetas
            true_idx = int(y_true_aligned[idx])
            pred_idx = int(y_pred_aligned[idx])
            
            true_label = task_label_names[true_idx] if true_idx < len(task_label_names) else f"Clase_{true_idx}"
            pred_label = task_label_names[pred_idx] if pred_idx < len(task_label_names) else f"Clase_{pred_idx}"
            
            # Confianza
            confidence = "N/A"
            if y_prob_aligned is not None and idx < len(y_prob_aligned):
                if y_prob_aligned.ndim == 2:
                    confidence = f"{np.max(y_prob_aligned[idx]):.3f}"
                else:
                    confidence = f"{y_prob_aligned[idx]:.3f}"
            
            is_correct = y_true_aligned[idx] == y_pred_aligned[idx]
            status = "✓" if is_correct else "✗"
            
            comparison = {
                'task': task,
                'sample_index': idx,
                'sample_text': sample_text,
                'true_label': true_label,
                'pred_label': pred_label,
                'confidence': confidence,
                'is_correct': is_correct,
                'status': status
            }
            
            comparisons.append(comparison)
        
        return comparisons
    
    def save_sample_comparisons_multitask(self, output_dir):
        """Guarda comparaciones muestra por muestra para todas las tareas y modelos"""
        logger.info("📝 Creando comparaciones muestra por muestra multitarea...")
        
        comparisons_dir = os.path.join(output_dir, 'sample_comparisons_multitask')
        os.makedirs(comparisons_dir, exist_ok=True)
        
        all_comparisons = {}
        
        for task in self.tasks:
            task_comparisons = {}
            task_dir = os.path.join(comparisons_dir, task)
            os.makedirs(task_dir, exist_ok=True)
            
            for model_name, results in self.results_by_task[task].items():
                if results['y_pred'] is None:
                    logger.warning(f"[{model_name}][{task}] Sin predicciones para comparar")
                    continue
                
                y_pred = results['y_pred']
                y_prob = results['y_prob']
                
                # Ground truth alineado
                y_true_aligned = self.convert_y_true_to_single_label(self.y_true)
                if len(y_true_aligned) != len(y_pred):
                    y_true_aligned = y_true_aligned[:len(y_pred)]
                
                # Crear comparaciones
                comparisons = self.create_sample_comparisons_multitask(
                    task, model_name, y_true_aligned, y_pred, y_prob, num_samples=10
                )
                
                task_comparisons[model_name] = comparisons
                
                # Archivos individuales
                model_display_name = results['metrics'].get('display_name', model_name)
                safe_model_name = "".join(c for c in model_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                
                # Archivo de texto
                txt_filename = os.path.join(task_dir, f'{safe_model_name}_comparisons.txt')
                self.save_model_comparison_txt_multitask(model_name, model_display_name, comparisons, txt_filename, task)
                
                # Archivo CSV
                csv_filename = os.path.join(task_dir, f'{safe_model_name}_comparisons.csv')
                df = pd.DataFrame(comparisons)
                df.to_csv(csv_filename, index=False, encoding='utf-8')
            
            all_comparisons[task] = task_comparisons
        
        # Archivo consolidado
        consolidated_file = os.path.join(comparisons_dir, 'all_tasks_models_comparisons.json')
        with open(consolidated_file, 'w', encoding='utf-8') as f:
            json_compatible = {}
            for task, task_comps in all_comparisons.items():
                json_compatible[task] = {}
                for model, comps in task_comps.items():
                    json_compatible[task][model] = []
                    for comp in comps:
                        json_comp = {}
                        for key, value in comp.items():
                            if isinstance(value, (np.integer, np.floating)):
                                json_comp[key] = float(value)
                            elif isinstance(value, np.bool_):
                                json_comp[key] = bool(value)
                            else:
                                json_comp[key] = value
                        json_compatible[task][model].append(json_comp)
            
            json.dump(json_compatible, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Comparaciones multitarea guardadas en {comparisons_dir}")
        return comparisons_dir
    
    def save_model_comparison_txt_multitask(self, model_name, model_display_name, comparisons, filename, task):
        """Guarda comparaciones de un modelo en formato texto para multitarea"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"COMPARACIONES MUESTRA POR MUESTRA - {model_display_name}\n")
            f.write(f"Modelo: {model_name}\n")
            f.write(f"Tarea: {task.replace('_', ' ').title()}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Formato: [Estado] [Muestra] -> [Predicción] vs [Real] (Confianza)\n\n")
            
            for i, comp in enumerate(comparisons, 1):
                f.write(f"{i:2d}. {comp['status']} [{comp['sample_index']:3d}] {comp['sample_text']}\n")
                f.write(f"     -> Predicción: {comp['pred_label']}\n")
                f.write(f"     -> Real:       {comp['true_label']}\n") 
                f.write(f"     -> Confianza:  {comp['confidence']}\n")
                f.write(f"     -> Correcto:   {'Sí' if comp['is_correct'] else 'No'}\n")
                f.write("\n")
            
            # Estadísticas
            correct_count = sum(1 for comp in comparisons if comp['is_correct'])
            total_count = len(comparisons)
            accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
            
            f.write("-" * 40 + "\n")
            f.write(f"ESTADÍSTICAS PARA {task.upper()} (muestra de {total_count} ejemplos):\n")
            f.write(f"Correctas: {correct_count}/{total_count} ({accuracy:.1f}%)\n")
            f.write(f"Incorrectas: {total_count - correct_count}/{total_count} ({100-accuracy:.1f}%)\n")
    
    def generate_multitask_report(self, output_dir):
        """Genera reporte completo multitarea"""
        logger.info("📊 Generando reporte híbrido multitarea...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Calcular métricas
        self.calculate_all_metrics_multitask()
        
        # Crear resumen de rendimiento
        summary_df = self.create_multitask_performance_summary()
        summary_df.to_csv(os.path.join(output_dir, 'multitask_performance_summary.csv'), index=False)
        
        # Guardar métricas detalladas
        detailed_metrics = {}
        for task in self.tasks:
            detailed_metrics[task] = {}
            for model_name, results in self.results_by_task[task].items():
                detailed_metrics[task][model_name] = results['metrics']
        
        with open(os.path.join(output_dir, 'multitask_detailed_metrics.json'), 'w') as f:
            json_compatible = {}
            for task, task_metrics in detailed_metrics.items():
                json_compatible[task] = {}
                for model, metrics in task_metrics.items():
                    json_compatible[task][model] = {}
                    for key, value in metrics.items():
                        if isinstance(value, np.ndarray):
                            json_compatible[task][model][key] = value.tolist()
                        elif isinstance(value, (np.float64, np.float32)):
                            json_compatible[task][model][key] = float(value)
                        else:
                            json_compatible[task][model][key] = value
            
            json.dump(json_compatible, f, indent=2)
        
        # Gráficos
        self.plot_multitask_comparison(os.path.join(output_dir, 'multitask_comparison.png'))
        
        # Matrices de confusión por tarea
        for task in self.tasks:
            self.plot_task_confusion_matrices(
                task, 
                os.path.join(output_dir, f'confusion_matrices_{task}.png')
            )
        
        # Comparaciones muestra por muestra
        comparisons_dir = self.save_sample_comparisons_multitask(output_dir)
        
        # Reporte de texto
        report_text = self.create_multitask_text_report(summary_df)
        with open(os.path.join(output_dir, 'multitask_benchmark_report.txt'), 'w') as f:
            f.write(report_text)
        
        logger.info(f"✅ Reporte multitarea guardado en {output_dir}")
        logger.info(f"📝 Comparaciones guardadas en {comparisons_dir}")
        return output_dir
    
    def create_multitask_text_report(self, summary_df):
        """Crea reporte de texto específico para multitarea"""
        report = []
        report.append("=" * 80)
        report.append("REPORTE DE BENCHMARKING - SISTEMA HÍBRIDO MULTITAREA")
        report.append("Ollama + Hugging Face + Scikit-learn")
        report.append("Con Equilibrio Automático de Clases")
        report.append("=" * 80)
        report.append("")
        
        # Información del experimento
        report.append("CONFIGURACIÓN DEL EXPERIMENTO:")
        report.append(f"  🎯 Enfoque: {self.metadata.get('approach', 'hybrid_multitask_system')}")
        report.append(f"  📋 Tareas: {', '.join([t.replace('_', ' ').title() for t in self.tasks])}")
        report.append(f"  🔢 Número de tareas: {len(self.tasks)}")
        report.append(f"  🤖 Modelos evaluados: {len(set(summary_df['Original_Name']))}")
        report.append(f"  📊 Total combinaciones: {len(summary_df)} (modelos × tareas)")
        report.append(f"  📈 Muestras de prueba: {len(self.y_true)}")
        report.append("")
        
        # Información de equilibrio de clases
        if self.balancing_info:
            report.append("EQUILIBRIO AUTOMÁTICO DE CLASES:")
            strategy = self.balancing_info.get('strategy', 'unknown')
            changes = self.balancing_info.get('changes', {})
            
            report.append(f"  ⚖️ Estrategia aplicada: {strategy}")
            report.append(f"  📊 Muestras originales: {self.balancing_info['original_stats']['total_samples']}")
            report.append(f"  📊 Muestras finales: {self.balancing_info['final_stats']['total_samples']}")
            
            if 'balance_improvement' in changes:
                improvement = changes['balance_improvement']
                report.append(f"  📈 Mejora en equilibrio: {improvement:.3f}")
            
            if self.balancing_info.get('excluded_classes'):
                excluded_count = len(self.balancing_info['excluded_classes'])
                report.append(f"  🚫 Clases excluidas: {excluded_count}")
            
            is_balanced = changes.get('is_now_balanced', False)
            report.append(f"  ✅ Ahora equilibrado: {'Sí' if is_balanced else 'No'}")
            report.append("")
        
        # Análisis por tarea
        report.append("ANÁLISIS POR TAREA:")
        for task in self.tasks:
            task_data = summary_df[summary_df['Task'] == task]
            task_name = task.replace('_', ' ').title()
            
            if len(task_data) > 0:
                best_model = task_data.iloc[0]
                avg_f1 = task_data['F1'].mean()
                
                report.append(f"  📋 {task_name}:")
                report.append(f"    Mejor modelo: {best_model['Model']} (F1: {best_model['F1']:.4f})")
                report.append(f"    F1 promedio: {avg_f1:.4f}")
                report.append(f"    Modelos evaluados: {len(task_data)}")
                
                # Etiquetas de la tarea
                task_info = self.task_metadata.get(task, {})
                num_labels = len(task_info.get('label_names', []))
                if num_labels > 0:
                    report.append(f"    Clases: {num_labels}")
                
                report.append("")
        
        # Análisis por backend
        report.append("ANÁLISIS POR BACKEND:")
        backend_counts = summary_df['Backend'].value_counts()
        
        for backend in ['ollama', 'huggingface', 'sklearn']:
            backend_data = summary_df[summary_df['Backend'] == backend]
            if len(backend_data) > 0:
                avg_f1 = backend_data['F1'].mean()
                best_overall = backend_data.loc[backend_data['F1'].idxmax()]
                
                report.append(f"  🔧 {backend.upper()}:")
                report.append(f"    Combinaciones: {len(backend_data)}")
                report.append(f"    F1 promedio: {avg_f1:.4f}")
                report.append(f"    Mejor: {best_overall['Model']} en {best_overall['Task']} ({best_overall['F1']:.4f})")
                report.append("")
        
        # Ranking general
        report.append("🏆 TOP 10 COMBINACIONES (Modelo + Tarea):")
        top_10 = summary_df.nlargest(10, 'F1')
        
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            task_name = row['Task'].replace('_', ' ').title()
            report.append(f"  {i:2d}. {row['Model']} en {task_name}")
            report.append(f"      Backend: {row['Backend']} | F1: {row['F1']:.4f} | Accuracy: {row['Accuracy']:.4f}")
        
        report.append("")
        
        # Análisis de velocidad
        if 'Samples_Per_Second' in summary_df.columns:
            speed_data = summary_df[summary_df['Samples_Per_Second'].notna()]
            
            if len(speed_data) > 0:
                report.append("⚡ ANÁLISIS DE VELOCIDAD:")
                fastest = speed_data.loc[speed_data['Samples_Per_Second'].idxmax()]
                slowest = speed_data.loc[speed_data['Samples_Per_Second'].idxmin()]
                
                report.append(f"  Más rápido: {fastest['Model']} ({fastest['Samples_Per_Second']:.2f} muestras/s)")
                report.append(f"  Más lento: {slowest['Model']} ({slowest['Samples_Per_Second']:.2f} muestras/s)")
                
                # Velocidad por backend
                for backend in ['ollama', 'huggingface', 'sklearn']:
                    backend_speed = speed_data[speed_data['Backend'] == backend]
                    if len(backend_speed) > 0:
                        avg_speed = backend_speed['Samples_Per_Second'].mean()
                        report.append(f"  {backend.title()} promedio: {avg_speed:.2f} muestras/s")
                
                report.append("")
        
        # Comparación entre tareas
        if len(self.tasks) > 1:
            report.append("📊 COMPARACIÓN ENTRE TAREAS:")
            
            for task in self.tasks:
                task_data = summary_df[summary_df['Task'] == task]
                if len(task_data) > 0:
                    task_avg = task_data['F1'].mean()
                    task_name = task.replace('_', ' ').title()
                    report.append(f"  {task_name}: F1 promedio = {task_avg:.4f}")
            
            # Modelos que funcionan bien en múltiples tareas
            model_performance = summary_df.groupby('Original_Name')['F1'].agg(['mean', 'count']).reset_index()
            multitask_models = model_performance[model_performance['count'] >= len(self.tasks)]
            
            if len(multitask_models) > 0:
                best_multitask = multitask_models.loc[multitask_models['mean'].idxmax()]
                model_display = summary_df[summary_df['Original_Name'] == best_multitask['Original_Name']]['Model'].iloc[0]
                
                report.append(f"  🎯 Mejor modelo multitarea: {model_display} (F1 promedio: {best_multitask['mean']:.4f})")
            
            report.append("")
        
        # Recomendaciones específicas
        report.append("💡 RECOMENDACIONES POR USO:")
        
        for task in self.tasks:
            task_data = summary_df[summary_df['Task'] == task]
            if len(task_data) > 0:
                best_accuracy = task_data.loc[task_data['Accuracy'].idxmax()]
                best_f1 = task_data.loc[task_data['F1'].idxmax()]
                
                task_name = task.replace('_', ' ').title()
                report.append(f"  📋 Para {task_name}:")
                report.append(f"    Máxima precisión: {best_f1['Model']} ({best_f1['Backend']})")
                report.append(f"    Máxima exactitud: {best_accuracy['Model']} ({best_accuracy['Backend']})")
        
        report.append("")
        
        # Conclusiones específicas del sistema híbrido multitarea
        report.append("🎯 CONCLUSIONES SISTEMA HÍBRIDO MULTITAREA:")
        report.append("  ✅ Evaluación simultánea de múltiples tareas de clasificación")
        report.append("  ✅ Equilibrio automático de clases para mejor rendimiento")
        report.append("  ✅ Comparación directa entre backends (Ollama, HuggingFace, Sklearn)")
        report.append("  ✅ Métricas específicas por tarea y combinadas")
        report.append("  ✅ Sistema 100% local y privado")
        report.append("  ✅ Escalable a nuevas tareas y modelos")
        report.append("")
        
        # Información sobre archivos generados
        report.append("📁 ARCHIVOS GENERADOS:")
        report.append("  📊 multitask_performance_summary.csv - Resumen de métricas por tarea")
        report.append("  📈 multitask_comparison.png - Gráfico comparativo multitarea")
        report.append("  🎨 confusion_matrices_[tarea].png - Matrices por tarea")
        report.append("  📝 sample_comparisons_multitask/ - Comparaciones por tarea")
        report.append("  🔧 multitask_detailed_metrics.json - Métricas detalladas")
        report.append("")
        
        report.append("💡 NOTA: Este reporte evalúa el rendimiento de cada modelo en cada tarea")
        report.append("   por separado, permitiendo identificar fortalezas específicas por dominio.")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def parse_console_arguments():
    """Parsea argumentos de línea de comandos para benchmarking multitarea"""
    parser = argparse.ArgumentParser(
        description='Benchmarking Multitarea para Sistema Híbrido con Equilibrio Automático',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Benchmarking multitarea básico
  python benchmarking_multitask_balanced.py --predictions ./predictions/hybrid_predictions.pkl
  
  # Con directorio de salida personalizado
  python benchmarking_multitask_balanced.py --predictions ./predictions/hybrid_predictions.pkl --output ./benchmarks/
  
  # Solo mostrar resumen sin generar archivos
  python benchmarking_multitask_balanced.py --predictions ./predictions/hybrid_predictions.pkl --summary-only
        """
    )
    
    parser.add_argument(
        '--predictions',
        type=str,
        default='./predictions/hybrid_predictions.pkl',
        help='Ruta al archivo de predicciones híbridas multitarea (.pkl)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./multitask_benchmark_results',
        help='Directorio de salida para el reporte (default: ./multitask_benchmark_results)'
    )
    
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Solo mostrar resumen en consola sin generar archivos'
    )
    
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=None,
        help='Tareas específicas a evaluar (default: todas las disponibles)'
    )
    
    parser.add_argument(
        '--colormap',
        type=str,
        default='Blues',
        choices=['Blues', 'viridis', 'plasma', 'coolwarm', 'RdYlBu'],
        help='Mapa de colores para matrices de confusión'
    )
    
    return parser.parse_args()


def run_multitask_benchmark(predictions_path, output_dir, tasks_filter=None, summary_only=False):
    """
    Ejecuta benchmarking para el sistema híbrido multitarea
    
    Args:
        predictions_path (str): Ruta al archivo de predicciones híbridas multitarea
        output_dir (str): Directorio de salida para el reporte
        tasks_filter (list): Lista de tareas específicas a evaluar (None = todas)
        summary_only (bool): Solo mostrar resumen sin generar archivos
    """
    # Crear benchmarker multitarea
    benchmarker = HybridModelBenchmarkMultitask(predictions_path)
    
    # Filtrar tareas si se especifica
    if tasks_filter:
        available_tasks = set(benchmarker.tasks)
        requested_tasks = set(tasks_filter)
        valid_tasks = requested_tasks.intersection(available_tasks)
        
        if not valid_tasks:
            logger.error(f"❌ Ninguna de las tareas solicitadas está disponible")
            logger.error(f"   Disponibles: {available_tasks}")
            logger.error(f"   Solicitadas: {requested_tasks}")
            return None
        
        if valid_tasks != requested_tasks:
            missing_tasks = requested_tasks - valid_tasks
            logger.warning(f"⚠️  Tareas no encontradas: {missing_tasks}")
        
        benchmarker.tasks = list(valid_tasks)
        logger.info(f"🎯 Evaluando tareas filtradas: {benchmarker.tasks}")
    
    if summary_only:
        # Solo calcular métricas y mostrar resumen
        benchmarker.calculate_all_metrics_multitask()
        summary = benchmarker.create_multitask_performance_summary()
        
        print("\n" + "="*80)
        print("RESUMEN DE RENDIMIENTO MULTITAREA")
        print("="*80)
        
        # Mostrar información básica
        print(f"📊 Tareas evaluadas: {', '.join(benchmarker.tasks)}")
        print(f"🤖 Modelos evaluados: {len(set(summary['Original_Name']))}")
        print(f"📈 Total combinaciones: {len(summary)}")
        
        # Equilibrio de clases
        if benchmarker.balancing_info:
            strategy = benchmarker.balancing_info.get('strategy', 'unknown')
            print(f"⚖️ Equilibrio aplicado: {strategy}")
        
        print("\n🏆 TOP 10 COMBINACIONES:")
        top_10 = summary.nlargest(10, 'F1')
        
        columns_to_show = ['Task', 'Model', 'Backend', 'F1', 'Accuracy', 'Precision', 'Recall']
        available_columns = [col for col in columns_to_show if col in summary.columns]
        
        print(top_10[available_columns].to_string(index=False))
        
        print("\n📊 RESUMEN POR TAREA:")
        for task in benchmarker.tasks:
            task_data = summary[summary['Task'] == task]
            if len(task_data) > 0:
                best_model = task_data.iloc[0]
                avg_f1 = task_data['F1'].mean()
                task_name = task.replace('_', ' ').title()
                
                print(f"  📋 {task_name}:")
                print(f"    Mejor: {best_model['Model']} (F1: {best_model['F1']:.4f})")
                print(f"    Promedio: {avg_f1:.4f}")
        
        print("\n💡 Para reporte completo ejecutar sin --summary-only")
        print("="*80)
        
        return summary
    
    else:
        # Generar reporte completo
        report_path = benchmarker.generate_multitask_report(output_dir)
        
        logger.info("✅ Benchmarking multitarea completado exitosamente")
        return report_path


# Función principal
if __name__ == "__main__":
    # Parsear argumentos
    args = parse_console_arguments()
    
    # Verificar que existe el archivo de predicciones
    if not os.path.exists(args.predictions):
        logger.error(f"❌ Archivo de predicciones no encontrado: {args.predictions}")
        logger.error("🔍 Verificar que se haya ejecutado hybrid_loader_balanced.py primero")
        exit(1)
    
    # Mostrar configuración
    logger.info("🚀 INICIANDO BENCHMARKING MULTITAREA")
    logger.info("=" * 60)
    logger.info(f"📁 Predicciones: {args.predictions}")
    logger.info(f"📊 Salida: {args.output}")
    logger.info(f"📋 Tareas: {args.tasks or 'Todas las disponibles'}")
    logger.info(f"🎨 Colormap: {args.colormap}")
    logger.info(f"📝 Solo resumen: {'Sí' if args.summary_only else 'No'}")
    logger.info("=" * 60)
    
    # Ejecutar benchmarking
    try:
        result = run_multitask_benchmark(
            predictions_path=args.predictions,
            output_dir=args.output,
            tasks_filter=args.tasks,
            summary_only=args.summary_only
        )
        
        if result is not None and not args.summary_only:
            print(f"\n🎉 Benchmarking multitarea completado exitosamente!")
            print(f"📁 Resultados guardados en: {args.output}")
            print(f"📊 Para ver resumen rápido usar: --summary-only")
        
    except Exception as e:
        logger.error(f"❌ Error durante benchmarking: {str(e)}")
        exit(1)