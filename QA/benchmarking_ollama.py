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

class OllamaGridSearchBenchmark:
    """
    Benchmarking especializado para resultados de Grid Search y RAG de Ollama
    Compatible con archivos pickle de ollama_gridsearch_predictor.py y rag_protein_classifier.py
    """
    def __init__(self, predictions_path):
        """
        Inicializa el benchmarking para modelos Ollama (Grid Search / RAG)
        
        Args:
            predictions_path (str): Ruta al archivo pickle con predicciones
        """
        self.predictions_path = predictions_path
        self.all_predictions = None
        self.sequences = None
        self.true_categories = None
        self.model_info = None
        self.standard_categories = None
        self.results = {}
        self.data_type = None  # 'gridsearch', 'rag', 'standard'
        
        self.load_predictions()
    
    def load_predictions(self):
        """Carga las predicciones desde el archivo pickle"""
        logger.info(f"🦙 Cargando predicciones de {self.predictions_path}...")
        
        with open(self.predictions_path, 'rb') as f:
            data = pickle.load(f)
        
        # Detectar formato del archivo
        if 'all_predictions' in data:
            # Formato de Grid Search / RAG
            self.all_predictions = data['all_predictions']
            self.sequences = data['sequences']
            self.true_categories = data['true_categories']
            self.model_info = data.get('model_info', {})
            self.standard_categories = data.get('standard_categories', [])
            
            # Detectar si es RAG o Grid Search
            sample_model = next(iter(self.all_predictions.values()))
            if isinstance(sample_model, dict) and 'predictions' in sample_model:
                if 'grid_search_score' in sample_model or 'optimized_params' in sample_model:
                    self.data_type = 'gridsearch'
                    logger.info("Detectado: Resultados de Grid Search")
                elif 'confidence' in sample_model or 'similar_proteins' in sample_model:
                    self.data_type = 'rag'
                    logger.info("🔍 Detectado: Resultados de RAG")
                else:
                    self.data_type = 'standard'
                    logger.info(" Ollama Detectado: Predicciones estándar")
            else:
                self.data_type = 'legacy'
                logger.info("Detectado: Formato legacy")
        
        # Formato alternativo (solo predicciones)
        elif isinstance(data, dict) and any('predictions' in str(k).lower() for k in data.keys()):
            self.all_predictions = data
            self.sequences = []
            self.true_categories = []
            self.model_info = {}
            self.standard_categories = []
            self.data_type = 'simple'
            logger.info("📊 Detectado: Formato simple de predicciones")
        
        else:
            raise ValueError(f"❌ Formato de archivo no reconocido. Estructura: {list(data.keys())}")
        
        # Información del dataset
        logger.info(f"✅ Datos cargados:")
        logger.info(f"   🤖 Modelos: {len(self.all_predictions)}")
        logger.info(f"   📊 Secuencias: {len(self.sequences)}")
        logger.info(f"   🎯 Categorías únicas: {len(set(self.true_categories)) if self.true_categories else 0}")
        logger.info(f"   📋 Categorías estándar: {len(self.standard_categories)}")
        logger.info(f"   📈 Tipo de datos: {self.data_type}")
        
        # Mostrar modelos disponibles
        logger.info(f"🦙 Modelos disponibles:")
        for model_name in self.all_predictions.keys():
            model_data = self.all_predictions[model_name]
            if isinstance(model_data, dict):
                model_type = model_data.get('model_type', 'unknown')
                family = model_data.get('model_family', model_data.get('family', 'unknown'))
                logger.info(f"   • {model_name} ({family}, {model_type})")
            else:
                logger.info(f"   • {model_name}")
    
    def extract_predictions_from_model(self, model_name, model_data):
        """Extrae predicciones de un modelo según el formato detectado"""
        
        if self.data_type == 'gridsearch':
            # Grid Search format
            if isinstance(model_data, dict) and 'predictions' in model_data:
                predictions = model_data['predictions']
                return predictions, model_data
            
        elif self.data_type == 'rag':
            # RAG format
            if isinstance(model_data, dict) and 'predictions' in model_data:
                predictions = model_data['predictions']
                return predictions, model_data
                
        elif self.data_type == 'standard' or self.data_type == 'legacy':
            # Standard format
            if isinstance(model_data, dict) and 'predictions' in model_data:
                predictions = model_data['predictions']
                return predictions, model_data
            elif isinstance(model_data, list):
                # Direct predictions list
                return model_data, {}
        
        elif self.data_type == 'simple':
            # Simple format - predicciones directas
            return model_data, {}
        
        return None, None
    
    def identify_model_info(self, model_name, model_data=None):
        """Identifica información del modelo"""
        info = {
            'family': 'Unknown',
            'backend': 'ollama',
            'model_type': 'unknown',
            'display_name': model_name,
            'original_name': model_name
        }
        
        # Información desde model_data
        if isinstance(model_data, dict):
            info['model_type'] = model_data.get('model_type', info['model_type'])
            info['family'] = model_data.get('model_family', model_data.get('family', info['family']))
            info['backend'] = model_data.get('backend', info['backend'])
            
            # Información adicional para Grid Search
            if self.data_type == 'gridsearch':
                info['optimized_params'] = model_data.get('optimized_params', {})
                info['grid_search_score'] = model_data.get('grid_search_score', np.nan)
                info['samples_per_second'] = model_data.get('samples_per_second', np.nan)
            
            # Información adicional para RAG
            elif self.data_type == 'rag':
                info['confidence_stats'] = model_data.get('confidence_stats', {})
                info['retrieval_stats'] = model_data.get('retrieval_stats', {})
        
        # Información desde model_info global
        if self.model_info and model_name in self.model_info:
            global_info = self.model_info[model_name]
            info['family'] = global_info.get('family', info['family'])
            info['model_type'] = global_info.get('type', info['model_type'])
        
        # Inferir familia desde nombre si no está disponible
        if info['family'] == 'Unknown':
            model_lower = model_name.lower()
            if 'deepseek' in model_lower:
                info['family'] = 'DeepSeek'
                if 'r1' in model_lower:
                    info['display_name'] = 'DeepSeek R1'
                elif '1.5b' in model_lower:
                    info['display_name'] = 'DeepSeek R1 1.5B'
                else:
                    info['display_name'] = 'DeepSeek LLM'
            elif 'llama' in model_lower:
                info['family'] = 'Llama'
                if '1b' in model_lower:
                    info['display_name'] = 'Llama 3.2 1B'
                elif '3b' in model_lower:
                    info['display_name'] = 'Llama 3.2 3B'
                else:
                    info['display_name'] = 'Llama 3.2'
            elif 'mistral' in model_lower:
                info['family'] = 'Mistral'
                info['display_name'] = 'Mistral'
            elif 'qwen' in model_lower:
                info['family'] = 'Qwen'
                if '2.5' in model_lower:
                    info['display_name'] = 'Qwen 2.5'
                else:
                    info['display_name'] = 'Qwen'
            else:
                info['family'] = model_name.split(':')[0].title()
                info['display_name'] = model_name.split(':')[0].title()
        
        return info
    
    def calculate_metrics(self, y_true, y_pred, model_name="unknown"):
        """Calcula métricas estándar"""
        metrics = {}
        
        try:
            # Convertir a numpy arrays
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # Validar dimensiones
            if len(y_true) != len(y_pred):
                min_len = min(len(y_true), len(y_pred))
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]
                logger.warning(f"[{model_name}] Ajustando dimensiones a {min_len}")
            
            # Métricas básicas
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
            # Precision, Recall, F1
            precision, recall, f1_score, support = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1_score'] = f1_score
            
            # Métricas macro
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0
            )
            metrics['precision_macro'] = precision_macro
            metrics['recall_macro'] = recall_macro
            metrics['f1_macro'] = f1_macro
            
            # Métricas por clase
            precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )
            metrics['precision_per_class'] = precision_per_class
            metrics['recall_per_class'] = recall_per_class
            metrics['f1_per_class'] = f1_per_class
            
            logger.info(f"[{model_name}] ✓ Métricas calculadas - Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}")
            
        except Exception as e:
            logger.error(f"[{model_name}] ❌ Error calculando métricas: {str(e)}")
            metrics = {
                'accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1_score': np.nan,
                'precision_macro': np.nan, 'recall_macro': np.nan, 'f1_macro': np.nan,
                'precision_per_class': np.array([]), 'recall_per_class': np.array([]),
                'f1_per_class': np.array([]), 'error': str(e)
            }
        
        return metrics
    
    def calculate_all_metrics(self):
        """Calcula métricas para todos los modelos"""
        logger.info("📊 Calculando métricas para todos los modelos...")
        
        for model_name, model_data in self.all_predictions.items():
            logger.info(f"🦙 Procesando {model_name}...")
            
            try:
                # Extraer predicciones
                predictions, extra_data = self.extract_predictions_from_model(model_name, model_data)
                
                if predictions is None:
                    logger.warning(f"[{model_name}] ❌ No se pudieron extraer predicciones")
                    continue
                
                # Calcular métricas
                metrics = self.calculate_metrics(self.true_categories, predictions, model_name)
                
                # Agregar información del modelo
                model_info = self.identify_model_info(model_name, extra_data)
                metrics.update(model_info)
                
                # Información adicional según tipo
                if isinstance(extra_data, dict):
                    metrics['prediction_time'] = extra_data.get('prediction_time', np.nan)
                    metrics['samples_per_second'] = extra_data.get('samples_per_second', np.nan)
                    
                    if self.data_type == 'gridsearch':
                        metrics['grid_search_metrics'] = extra_data.get('grid_search_metrics', {})
                    elif self.data_type == 'rag':
                        metrics['rag_statistics'] = extra_data.get('statistics', {})
                
                # Guardar resultados
                self.results[model_name] = {
                    'metrics': metrics,
                    'predictions': predictions,
                    'model_data': extra_data
                }
                
                logger.info(f"[{model_name}] ✅ Procesado exitosamente")
                
            except Exception as e:
                logger.error(f"[{model_name}] ❌ Error: {str(e)}")
                continue
        
        logger.info(f"✅ Métricas calculadas para {len(self.results)} modelos")
    
    def create_performance_summary(self):
        """Crea resumen de rendimiento"""
        logger.info("📊 Creando resumen de rendimiento...")
        
        summary_data = []
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            
            row = {
                'Model': metrics.get('display_name', model_name),
                'Original_Name': model_name,
                'Family': metrics.get('family', 'Unknown'),
                'Model_Type': metrics.get('model_type', 'unknown'),
                'Backend': metrics.get('backend', 'ollama'),
                'Data_Type': self.data_type,
                'accuracy': metrics.get('accuracy', np.nan),
                'f1_score': metrics.get('f1_score', np.nan),
                'precision': metrics.get('precision', np.nan),
                'recall': metrics.get('recall', np.nan),
                'f1_macro': metrics.get('f1_macro', np.nan),
                'precision_macro': metrics.get('precision_macro', np.nan),
                'recall_macro': metrics.get('recall_macro', np.nan),
                'prediction_time': metrics.get('prediction_time', np.nan),
                'samples_per_second': metrics.get('samples_per_second', np.nan)
            }
            
            # Información específica según tipo
            if self.data_type == 'gridsearch':
                row['grid_search_score'] = metrics.get('grid_search_score', np.nan)
                row['optimized_params'] = str(metrics.get('optimized_params', {}))
            elif self.data_type == 'rag':
                rag_stats = metrics.get('rag_statistics', {})
                row['avg_confidence'] = rag_stats.get('avg_confidence', np.nan)
                row['avg_retrieval_time'] = rag_stats.get('avg_retrieval_time', np.nan)
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('f1_score', ascending=False, na_position='last')
        
        return summary_df
    
    def plot_performance_comparison(self, save_path=None):
        """Crea gráfico de comparación de rendimiento"""
        summary_df = self.create_performance_summary()
        
        # Colores por familia
        family_colors = {
            'DeepSeek': '#FF6B35',
            'Llama': '#4285F4', 
            'Mistral': '#34A853',
            'Qwen': '#9C27B0',
            'Gemma': '#FF9800',
            'Phi': '#E91E63',
            'Unknown': '#9E9E9E'
        }
        
        # Métricas a mostrar
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        available_metrics = [m for m in metrics if m in summary_df.columns and summary_df[m].notna().any()]
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            data = summary_df[['Model', 'Family', metric]].dropna()
            
            if len(data) == 0:
                continue
            
            # Crear barras coloreadas por familia
            bars = ax.bar(range(len(data)), data[metric], 
                         color=[family_colors.get(f, '#gray') for f in data['Family']])
            
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
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
        
        # Leyenda
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=family) 
                          for family, color in family_colors.items() 
                          if family in summary_df['Family'].values]
        
        if legend_elements:
            fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                      ncol=len(legend_elements), title='Familia de Modelo')
        
        title = f'Comparación de Rendimiento - {self.data_type.title()}'
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado en {save_path}")
        
        plt.show()
        return fig
    
    def plot_confusion_matrix(self, model_name=None, save_path=None):
        """Crea matriz de confusión para un modelo específico"""
        if model_name is None:
            # Usar el mejor modelo
            summary_df = self.create_performance_summary()
            model_name = summary_df.iloc[0]['Original_Name']
        
        if model_name not in self.results:
            logger.error(f"Modelo {model_name} no encontrado")
            return None
        
        predictions = self.results[model_name]['predictions']
        y_true = np.array(self.true_categories)
        y_pred = np.array(predictions)
        
        # Ajustar dimensiones si es necesario
        if len(y_true) != len(y_pred):
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
        
        # Obtener etiquetas únicas
        labels = sorted(list(set(y_true) | set(y_pred)))
        
        # Calcular matriz de confusión
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plotear
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        
        model_display = self.results[model_name]['metrics'].get('display_name', model_name)
        plt.title(f'Matriz de Confusión - {model_display}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicción')
        plt.ylabel('Verdadero')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Matriz de confusión guardada en {save_path}")
        
        plt.show()
        return plt.gcf()
    
    def generate_report(self, output_dir):
        """Genera reporte completo"""
        logger.info("📊 Generando reporte completo...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Calcular métricas
        self.calculate_all_metrics()
        
        # Crear resumen
        summary_df = self.create_performance_summary()
        summary_df.to_csv(os.path.join(output_dir, 'performance_summary.csv'), index=False)
        
        # Gráficos
        self.plot_performance_comparison(os.path.join(output_dir, 'performance_comparison.png'))
        
        # Matriz de confusión del mejor modelo
        if len(self.results) > 0:
            best_model = summary_df.iloc[0]['Original_Name']
            self.plot_confusion_matrix(best_model, os.path.join(output_dir, f'confusion_matrix_{best_model}.png'))
        
        # Reporte de texto
        report_text = self.create_text_report(summary_df)
        with open(os.path.join(output_dir, 'benchmark_report.txt'), 'w') as f:
            f.write(report_text)
        
        # Guardar métricas detalladas
        detailed_metrics = {}
        for model_name, result in self.results.items():
            detailed_metrics[model_name] = result['metrics']
        
        with open(os.path.join(output_dir, 'detailed_metrics.json'), 'w') as f:
            # Convertir numpy arrays a listas para JSON
            json_data = {}
            for model, metrics in detailed_metrics.items():
                json_data[model] = {}
                for key, value in metrics.items():
                    if isinstance(value, np.ndarray):
                        json_data[model][key] = value.tolist()
                    elif isinstance(value, (np.float64, np.float32)):
                        json_data[model][key] = float(value)
                    else:
                        json_data[model][key] = value
            
            json.dump(json_data, f, indent=2)
        
        logger.info(f"✅ Reporte guardado en {output_dir}")
        return output_dir
    
    def create_text_report(self, summary_df):
        """Crea reporte de texto"""
        report = []
        report.append("=" * 80)
        report.append("REPORTE DE BENCHMARKING - MODELOS OLLAMA")
        report.append("=" * 80)
        report.append("")
        
        # Información general
        report.append("CONFIGURACIÓN DEL EXPERIMENTO:")
        report.append(f"  📁 Archivo: {self.predictions_path}")
        report.append(f"  📊 Tipo de datos: {self.data_type}")
        report.append(f"  🤖 Modelos evaluados: {len(self.results)}")
        report.append(f"  📈 Secuencias: {len(self.sequences)}")
        report.append(f"  🎯 Categorías: {len(set(self.true_categories)) if self.true_categories else 0}")
        report.append("")
        
        # Top 10 modelos
        report.append("🏆 TOP 10 MODELOS:")
        top_10 = summary_df.head(10)
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            report.append(f"  {i:2d}. {row['Model']} ({row['Family']})")
            report.append(f"      F1: {row['f1_score']:.4f} | Accuracy: {row['accuracy']:.4f}")
        report.append("")
        
        # Análisis por familia
        report.append("📊 ANÁLISIS POR FAMILIA:")
        families = summary_df.groupby('Family').agg({
            'f1_score': ['mean', 'std', 'count'],
            'accuracy': ['mean', 'std']
        }).round(4)
        
        for family in families.index:
            if family != 'Unknown':
                count = families.loc[family, ('f1_score', 'count')]
                f1_mean = families.loc[family, ('f1_score', 'mean')]
                f1_std = families.loc[family, ('f1_score', 'std')]
                acc_mean = families.loc[family, ('accuracy', 'mean')]
                
                report.append(f"  {family}: {count} modelos")
                report.append(f"    F1 promedio: {f1_mean:.4f} (±{f1_std:.4f})")
                report.append(f"    Accuracy promedio: {acc_mean:.4f}")
        report.append("")
        
        # Información específica según tipo
        if self.data_type == 'gridsearch':
            report.append("🔍 INFORMACIÓN GRID SEARCH:")
            gs_models = summary_df[summary_df['grid_search_score'].notna()]
            if len(gs_models) > 0:
                best_gs = gs_models.loc[gs_models['grid_search_score'].idxmax()]
                report.append(f"  Mejor score de optimización: {best_gs['grid_search_score']:.4f}")
                report.append(f"  Modelo: {best_gs['Model']}")
            report.append("")
        
        elif self.data_type == 'rag':
            report.append("🔍 INFORMACIÓN RAG:")
            rag_models = summary_df[summary_df['avg_confidence'].notna()]
            if len(rag_models) > 0:
                avg_conf = rag_models['avg_confidence'].mean()
                report.append(f"  Confianza promedio: {avg_conf:.4f}")
            report.append("")
        
        # Velocidad
        speed_data = summary_df[summary_df['samples_per_second'].notna()]
        if len(speed_data) > 0:
            report.append("⚡ ANÁLISIS DE VELOCIDAD:")
            fastest = speed_data.loc[speed_data['samples_per_second'].idxmax()]
            slowest = speed_data.loc[speed_data['samples_per_second'].idxmin()]
            
            report.append(f"  Más rápido: {fastest['Model']} ({fastest['samples_per_second']:.2f} seq/s)")
            report.append(f"  Más lento: {slowest['Model']} ({slowest['samples_per_second']:.2f} seq/s)")
            report.append("")
        
        # Recomendaciones
        if len(summary_df) > 0:
            best_overall = summary_df.iloc[0]
            report.append("💡 RECOMENDACIONES:")
            report.append(f"  🏆 Mejor modelo general: {best_overall['Model']}")
            report.append(f"    F1 Score: {best_overall['f1_score']:.4f}")
            report.append(f"    Accuracy: {best_overall['accuracy']:.4f}")
            report.append(f"    Familia: {best_overall['Family']}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def parse_arguments():
    """Parsea argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='Benchmarking para resultados de Ollama (Grid Search / RAG)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Benchmarking básico
  python benchmarking_ollama_fixed.py --predictions cog_predictions.pkl
  
  # Con directorio de salida personalizado
  python benchmarking_ollama_fixed.py --predictions cog_predictions.pkl --output ./results/
  
  # Solo mostrar resumen
  python benchmarking_ollama_fixed.py --predictions cog_predictions.pkl --summary-only
        """
    )
    
    parser.add_argument(
        '--predictions',
        type=str,
        default='cog_predictions.pkl',
        help='Ruta al archivo de predicciones (.pkl)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./benchmark_results',
        help='Directorio de salida para el reporte'
    )
    
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Solo mostrar resumen en consola sin generar archivos'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Modelo específico para matriz de confusión (default: mejor modelo)'
    )
    
    return parser.parse_args()


def run_benchmark(predictions_path, output_dir, summary_only=False, specific_model=None):
    """
    Ejecuta benchmarking completo
    
    Args:
        predictions_path (str): Ruta al archivo de predicciones
        output_dir (str): Directorio de salida
        summary_only (bool): Solo mostrar resumen
        specific_model (str): Modelo específico para análisis detallado
    """
    # Crear benchmarker
    benchmarker = OllamaGridSearchBenchmark(predictions_path)
    
    if summary_only:
        # Solo calcular métricas y mostrar resumen
        benchmarker.calculate_all_metrics()
        summary = benchmarker.create_performance_summary()
        
        print("\n" + "="*80)
        print("RESUMEN DE RENDIMIENTO - MODELOS OLLAMA")
        print("="*80)
        
        # Información básica
        print(f"📊 Tipo de datos: {benchmarker.data_type}")
        print(f"🤖 Modelos evaluados: {len(benchmarker.results)}")
        print(f"📈 Secuencias: {len(benchmarker.sequences)}")
        print(f"🎯 Categorías: {len(set(benchmarker.true_categories)) if benchmarker.true_categories else 0}")
        
        print(f"\n🏆 TOP 10 MODELOS:")
        print("-" * 80)
        
        # Mostrar columnas más importantes
        display_columns = ['Model', 'Family', 'accuracy', 'f1_score', 'precision', 'recall']
        available_columns = [col for col in display_columns if col in summary.columns]
        
        if len(summary) > 0:
            top_10 = summary.head(10)
            print(top_10[available_columns].to_string(index=False, float_format='%.4f'))
        else:
            print("No hay resultados para mostrar")
        
        # Análisis por familia
        if len(summary) > 0 and 'Family' in summary.columns:
            print(f"\n📊 RESUMEN POR FAMILIA:")
            print("-" * 40)
            family_stats = summary.groupby('Family').agg({
                'f1_score': ['count', 'mean', 'std'],
                'accuracy': 'mean'
            }).round(4)
            
            for family in family_stats.index:
                if family != 'Unknown':
                    count = family_stats.loc[family, ('f1_score', 'count')]
                    f1_mean = family_stats.loc[family, ('f1_score', 'mean')]
                    f1_std = family_stats.loc[family, ('f1_score', 'std')]
                    acc_mean = family_stats.loc[family, ('accuracy', 'mean')]
                    
                    print(f"  {family}: {count} modelos")
                    print(f"    F1: {f1_mean:.4f} (±{f1_std:.4f}) | Accuracy: {acc_mean:.4f}")
        
        # Información específica del tipo de datos
        if benchmarker.data_type == 'gridsearch':
            print(f"\n🔍 GRID SEARCH INFO:")
            gs_info = summary[summary['grid_search_score'].notna()]
            if len(gs_info) > 0:
                best_gs = gs_info.iloc[0]
                print(f"  Mejor optimización: {best_gs['Model']} (Score: {best_gs['grid_search_score']:.4f})")
        
        elif benchmarker.data_type == 'rag':
            print(f"\n🔍 RAG INFO:")
            rag_info = summary[summary['avg_confidence'].notna()]
            if len(rag_info) > 0:
                avg_conf = rag_info['avg_confidence'].mean()
                print(f"  Confianza promedio: {avg_conf:.4f}")
        
        print("\n💡 Para reporte completo ejecutar sin --summary-only")
        print("="*80)
        
        return summary
    
    else:
        # Generar reporte completo
        report_path = benchmarker.generate_report(output_dir)
        
        # Mostrar matriz de confusión del modelo específico si se solicita
        if specific_model and specific_model in benchmarker.results:
            benchmarker.plot_confusion_matrix(
                specific_model, 
                os.path.join(output_dir, f'confusion_matrix_{specific_model}_detailed.png')
            )
        
        logger.info("✅ Benchmarking completado exitosamente")
        return report_path


def main():
    """Función principal"""
    args = parse_arguments()
    
    # Verificar que existe el archivo
    if not os.path.exists(args.predictions):
        logger.error(f"❌ Archivo no encontrado: {args.predictions}")
        logger.error("🔍 Verificar la ruta del archivo de predicciones")
        exit(1)
    
    # Mostrar configuración
    logger.info("🦙 INICIANDO BENCHMARKING OLLAMA")
    logger.info("=" * 50)
    logger.info(f"📁 Predicciones: {args.predictions}")
    logger.info(f"📊 Salida: {args.output}")
    logger.info(f"📝 Solo resumen: {'Sí' if args.summary_only else 'No'}")
    if args.model:
        logger.info(f"🎯 Modelo específico: {args.model}")
    logger.info("=" * 50)
    
    # Ejecutar benchmarking
    try:
        result = run_benchmark(
            predictions_path=args.predictions,
            output_dir=args.output,
            summary_only=args.summary_only,
            specific_model=args.model
        )
        
        if result is not None and not args.summary_only:
            print(f"\n🎉 Benchmarking completado exitosamente!")
            print(f"📁 Resultados guardados en: {args.output}")
            print(f"📊 Para ver resumen rápido usar: --summary-only")
        
    except Exception as e:
        logger.error(f"❌ Error durante benchmarking: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
