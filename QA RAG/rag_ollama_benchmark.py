import os
import pandas as pd
import numpy as np
import requests
import json
import time
import re
import argparse
import sys
import sqlite3
import faiss
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import logging
from dataclasses import dataclass
from collections import defaultdict, Counter
from datetime import datetime

# Para embeddings de proteínas
import torch
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Resultado de benchmark para una secuencia"""
    sequence_id: int
    sequence: str
    true_category: str
    true_group: str
    rag_prediction: str
    rag_confidence: float
    ollama_prediction: str
    ollama_confidence: str
    ollama_analysis: str
    ensemble_prediction: str
    ensemble_confidence: float
    processing_time: float
    prompt_tokens: int
    response_tokens: int
    rag_evidence: List[Dict]
    raw_ollama_response: str

class ProteinEmbedder:
    """Clase para generar embeddings de proteínas usando modelos pre-entrenados"""
    
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D", device="gpu"):
        self.device = device
        self.model_name = model_name
        
        logger.info(f"🧬 Cargando modelo de embeddings: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            
            # Obtener dimensión de embeddings
            with torch.no_grad():
                test_input = self.tokenizer("M", return_tensors="pt").to(device)
                test_output = self.model(**test_input)
                self.embedding_dim = test_output.last_hidden_state.shape[-1]
            
            logger.info(f"✅ Modelo cargado. Dimensión: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise e
    
    def encode(self, sequences: List[str]) -> np.ndarray:
        """Genera embeddings para una lista de secuencias"""
        embeddings = []
        
        with torch.no_grad():
            for sequence in sequences:
                # Truncar secuencias muy largas
                seq = sequence[:1000] if len(sequence) > 1000 else sequence
                
                try:
                    inputs = self.tokenizer(
                        seq, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True,
                        max_length=1022
                    ).to(self.device)
                    
                    outputs = self.model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.extend(embedding)
                    
                except Exception as e:
                    logger.warning(f"Error procesando secuencia: {e}")
                    embeddings.append(np.zeros(self.embedding_dim))
        
        return np.array(embeddings)

class COGRAGRetriever:
    """Sistema de recuperación para el RAG COG"""
    
    def __init__(self, rag_path: str = "RAG_COG"):
        self.rag_path = Path(rag_path)
        
        if not self.rag_path.exists():
            raise FileNotFoundError(f"No se encontró el directorio RAG: {rag_path}")
        
        logger.info(f"🔍 Cargando RAG COG desde: {self.rag_path}")
        
        # Cargar configuración
        config_path = self.rag_path / "config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Cargar índice FAISS
        faiss_path = str(self.rag_path / "cog_rag.index")
        self.index = faiss.read_index(faiss_path)
        
        # Conectar a base de datos
        db_path = self.rag_path / "cog_rag.db"
        self.conn = sqlite3.connect(db_path)
        
        # Inicializar embedder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = ProteinEmbedder(
            model_name=self.config['model_name'], 
            device=device
        )
        
        logger.info(f"✅ RAG COG cargado: {self.config['total_entries']:,} entradas")
    
    def get_random_test_sequences(self, n_samples: int, exclude_ids: List[int] = None) -> List[Dict]:
        """Obtiene secuencias aleatorias del RAG para testing"""
        logger.info(f"🎲 Seleccionando {n_samples} secuencias aleatorias para testing...")
        
        # Obtener IDs disponibles
        cursor = self.conn.execute("SELECT COUNT(*) FROM cog_entries")
        total_entries = cursor.fetchone()[0]
        
        # Generar IDs aleatorios
        available_ids = list(range(total_entries))
        if exclude_ids:
            available_ids = [id for id in available_ids if id not in exclude_ids]
        
        if len(available_ids) < n_samples:
            logger.warning(f"⚠️ Solo hay {len(available_ids)} secuencias disponibles, usando todas")
            n_samples = len(available_ids)
        
        selected_ids = random.sample(available_ids, n_samples)
        
        # Obtener secuencias seleccionadas
        test_sequences = []
        for seq_id in selected_ids:
            cursor = self.conn.execute(
                "SELECT * FROM cog_entries WHERE id = ?", (seq_id,)
            )
            row = cursor.fetchone()
            
            if row:
                test_sequences.append({
                    'id': row[0],
                    'protein_id': row[1],
                    'sequence': row[2],
                    'functional_region': row[3],
                    'cog_id': row[4],
                    'category_id': row[5],
                    'category_description': row[6],
                    'functional_group': row[7],
                    'organism': row[13]
                })
        
        logger.info(f"✅ Seleccionadas {len(test_sequences)} secuencias de testing")
        
        # Mostrar distribución por categoría
        categories = [seq['category_id'] for seq in test_sequences]
        category_counts = Counter(categories)
        logger.info(f"📊 Distribución por categoría:")
        for cat, count in sorted(category_counts.items()):
            desc = self.config['categories'].get(cat, 'Unknown')
            logger.info(f"   {cat}: {count} secuencias ({desc})")
        
        return test_sequences
    
    def predict_category(self, query_sequence: str, k: int = 10, exclude_protein_id: str = None) -> Dict:
        """Predice la categoría COG usando RAG, excluyendo la proteína de consulta"""
        # Generar embedding de la consulta
        query_embedding = self.embedder.encode([query_sequence])
        
        # Buscar en el índice
        distances, indices = self.index.search(
            query_embedding.astype(np.float32), k * 5  # Buscar más para filtrar
        )
        
        # Convertir distancias a similitudes
        similarities = 1 / (1 + distances[0])
        
        # Recuperar información de la base de datos, excluyendo la proteína de consulta
        similar_sequences = []
        for idx, similarity in zip(indices[0], similarities):
            if similarity >= 0.3:  # Umbral mínimo
                cursor = self.conn.execute(
                    "SELECT * FROM cog_entries WHERE id = ?", (int(idx),)
                )
                row = cursor.fetchone()
                
                if row and row[1] != exclude_protein_id:  # Excluir la proteína de consulta
                    similar_sequences.append({
                        'similarity': float(similarity),
                        'protein_id': row[1],
                        'cog_id': row[4],
                        'category_id': row[5],
                        'category_description': row[6],
                        'functional_group': row[7],
                        'organism': row[13]
                    })
                    
                    if len(similar_sequences) >= k:
                        break
        
        if not similar_sequences:
            return {
                'predicted_category': 'S',
                'category_description': 'Function unknown',
                'functional_group': 'POORLY CHARACTERIZED',
                'confidence': 0.0,
                'evidence': [],
                'category_scores': {'S': 1.0},
                'method': 'RAG'
            }
        
        # Votar por categoría ponderado por similitud
        category_votes = defaultdict(float)
        total_weight = 0
        
        for seq_info in similar_sequences:
            weight = seq_info['similarity']
            category = seq_info['category_id']
            category_votes[category] += weight
            total_weight += weight
        
        # Normalizar scores
        category_scores = {
            cat: score / total_weight 
            for cat, score in category_votes.items()
        }
        
        # Predicción final
        predicted_category = max(category_scores.keys(), key=category_scores.get)
        confidence = category_scores[predicted_category]
        
        return {
            'predicted_category': predicted_category,
            'category_description': self.config['categories'].get(predicted_category, 'Unknown'),
            'functional_group': self.config['functional_groups'].get(predicted_category, 'Unknown'),
            'confidence': float(confidence),
            'evidence': similar_sequences,
            'category_scores': {k: float(v) for k, v in category_scores.items()},
            'method': 'RAG'
        }

class OllamaClient:
    """Cliente para interactuar con Ollama API"""
    
    def __init__(self, host="localhost", port=11434, timeout=120):
        self.base_url = f"http://{host}:{port}"
        self.api_url = f"{self.base_url}/api"
        self.timeout = timeout
        self.available_models = self._get_available_models()
        
        logger.info(f"🦙 Cliente Ollama inicializado: {self.base_url}")
        logger.info(f"⏱️ Timeout: {timeout}s")
    
    def _get_available_models(self) -> List[str]:
        """Obtiene lista de modelos disponibles en Ollama"""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                logger.info(f"📊 Modelos disponibles: {', '.join(models)}")
                return models
            return []
        except Exception as e:
            logger.warning(f"⚠️ Error conectando con Ollama: {e}")
            return []
    
    def is_model_available(self, model_name: str) -> bool:
        """Verifica si un modelo específico está disponible"""
        return model_name in self.available_models
    
    def generate_text(self, model: str, prompt: str, **kwargs) -> Optional[Dict]:
        """Genera texto usando Ollama"""
        if not self.is_model_available(model):
            logger.error(f"❌ Modelo {model} no disponible")
            logger.info(f"📋 Modelos disponibles: {', '.join(self.available_models)}")
            return None
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        try:
            start_time = time.time()
            response = requests.post(f"{self.api_url}/generate", json=data, timeout=self.timeout)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                result['response_time'] = end_time - start_time
                result['prompt_eval_count'] = result.get('prompt_eval_count', 0)
                result['eval_count'] = result.get('eval_count', 0)
                return result
            else:
                logger.error(f"❌ Error Ollama {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error en generación: {e}")
            return None

class EnhancedCOGPromptGenerator:
    """Generador de prompts mejorado que incluye evidencia del RAG"""
    
    def __init__(self):
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
        
        self.functional_groups = {
            'J': 'INFORMATION STORAGE AND PROCESSING',
            'A': 'INFORMATION STORAGE AND PROCESSING',
            'K': 'INFORMATION STORAGE AND PROCESSING',
            'L': 'INFORMATION STORAGE AND PROCESSING',
            'B': 'INFORMATION STORAGE AND PROCESSING',
            'D': 'CELLULAR PROCESSES AND SIGNALING',
            'O': 'CELLULAR PROCESSES AND SIGNALING',
            'M': 'CELLULAR PROCESSES AND SIGNALING',
            'N': 'CELLULAR PROCESSES AND SIGNALING',
            'P': 'CELLULAR PROCESSES AND SIGNALING',
            'T': 'CELLULAR PROCESSES AND SIGNALING',
            'U': 'CELLULAR PROCESSES AND SIGNALING',
            'V': 'CELLULAR PROCESSES AND SIGNALING',
            'W': 'CELLULAR PROCESSES AND SIGNALING',
            'C': 'METABOLISM',
            'G': 'METABOLISM',
            'E': 'METABOLISM',
            'F': 'METABOLISM',
            'H': 'METABOLISM',
            'I': 'METABOLISM',
            'Q': 'METABOLISM',
            'R': 'POORLY CHARACTERIZED',
            'S': 'POORLY CHARACTERIZED'
        }
    
    def generate_enhanced_prompt(self, sequence: str, rag_evidence: List[Dict] = None, 
                               include_evidence: bool = True, max_evidence: int = 3) -> str:
        """Genera prompt mejorado con evidencia del RAG opcional"""
        
        base_prompt = f"""You are an expert protein function prediction system. Analyze the given protein sequence and predict its COG functional category and group.

COG CATEGORIES AND GROUPS:

INFORMATION STORAGE AND PROCESSING:
- J: Translation, ribosomal structure and biogenesis
- A: RNA processing and modification
- K: Transcription
- L: Replication, recombination and repair
- B: Chromatin structure and dynamics

CELLULAR PROCESSES AND SIGNALING:
- D: Cell cycle control, cell division
- O: Molecular chaperones and related functions
- M: Cell wall/membrane/envelope biogenesis
- N: Cell motility
- P: Inorganic ion transport and metabolism
- T: Signal transduction mechanisms
- U: Intracellular trafficking, secretion
- V: Defense mechanisms
- W: Extracellular structures

METABOLISM:
- C: Energy production and conversion
- G: Carbohydrate transport and metabolism
- E: Amino acid transport and metabolism
- F: Nucleotide transport and metabolism
- H: Coenzyme transport and metabolism
- I: Lipid transport and metabolism
- Q: Secondary metabolites biosynthesis

POORLY CHARACTERIZED:
- R: General function prediction only
- S: Function unknown

PROTEIN TO ANALYZE:
Sequence: {sequence}
Length: {len(sequence)} amino acids"""

        # Agregar evidencia del RAG si está disponible y se solicita
        if include_evidence and rag_evidence and len(rag_evidence) > 0:
            base_prompt += f"\n\nSIMILAR PROTEINS FOUND (for reference):"
            for i, evidence in enumerate(rag_evidence[:max_evidence], 1):
                base_prompt += f"""
Evidence {i}:
- Similarity: {evidence['similarity']:.3f}
- Category: {evidence['category_id']} ({evidence['category_description']})
- Group: {evidence['functional_group']}
- Organism: {evidence['organism']}
- COG ID: {evidence['cog_id']}"""

        base_prompt += f"""

ANALYSIS INSTRUCTIONS:
1. Examine the amino acid sequence for characteristic patterns and motifs
2. Consider the sequence length and composition
3. {"Use similar proteins as supporting evidence" if include_evidence and rag_evidence else "Make prediction based on sequence analysis alone"}
4. Provide your best prediction with reasoning

REQUIRED OUTPUT FORMAT:
Analysis: [Brief explanation of your reasoning, including any relevant sequence features]
Category: [Single letter: J,A,K,L,B,D,O,M,N,P,T,U,V,W,C,G,E,F,H,I,Q,R,S]
Group: [Full group name: INFORMATION STORAGE AND PROCESSING, CELLULAR PROCESSES AND SIGNALING, METABOLISM, or POORLY CHARACTERIZED]
Confidence: [High/Medium/Low]"""

        return base_prompt
    
    def extract_prediction_from_response(self, response: str) -> Dict:
        """Extrae la predicción estructurada de la respuesta del LLM"""
        response = response.strip()
        
        # Buscar categoría
        category_match = re.search(r'Category:\s*([A-S])\b', response, re.IGNORECASE)
        category = category_match.group(1).upper() if category_match else 'S'
        
        # Buscar grupo
        group_patterns = [
            r'Group:\s*(INFORMATION STORAGE AND PROCESSING|CELLULAR PROCESSES AND SIGNALING|METABOLISM|POORLY CHARACTERIZED)',
            r'Group:\s*(Information Storage and Processing|Cellular Processes and Signaling|Metabolism|Poorly Characterized)'
        ]
        
        group = 'POORLY CHARACTERIZED'  # Default
        for pattern in group_patterns:
            group_match = re.search(pattern, response, re.IGNORECASE)
            if group_match:
                group = group_match.group(1).upper()
                break
        
        # Buscar confianza
        confidence_match = re.search(r'Confidence:\s*(High|Medium|Low)', response, re.IGNORECASE)
        confidence = confidence_match.group(1).title() if confidence_match else 'Low'
        
        # Extraer análisis
        analysis_match = re.search(r'Analysis:\s*([^\n]+(?:\n[^Category]+)*)', response, re.IGNORECASE)
        analysis = analysis_match.group(1).strip() if analysis_match else "No analysis provided"
        
        return {
            'predicted_category': category,
            'category_description': self.cog_categories.get(category, 'Unknown'),
            'functional_group': group,
            'confidence': confidence,
            'analysis': analysis,
            'raw_response': response,
            'method': 'Ollama'
        }

class RAGOllamaBenchmark:
    """Sistema de benchmark para evaluar RAG + Ollama con secuencias COG"""
    
    def __init__(self, rag_path: str = "RAG_COG", ollama_host: str = "localhost", 
                 ollama_port: int = 11434, timeout: int = 120):
        
        # Inicializar componentes
        self.rag_retriever = COGRAGRetriever(rag_path)
        self.ollama_client = OllamaClient(ollama_host, ollama_port, timeout)
        self.prompt_generator = EnhancedCOGPromptGenerator()
        
        # Mapeo de grupos funcionales
        self.group_mapping = {
            'J': 'INFORMATION STORAGE AND PROCESSING',
            'A': 'INFORMATION STORAGE AND PROCESSING',
            'K': 'INFORMATION STORAGE AND PROCESSING',
            'L': 'INFORMATION STORAGE AND PROCESSING',
            'B': 'INFORMATION STORAGE AND PROCESSING',
            'D': 'CELLULAR PROCESSES AND SIGNALING',
            'O': 'CELLULAR PROCESSES AND SIGNALING',
            'M': 'CELLULAR PROCESSES AND SIGNALING',
            'N': 'CELLULAR PROCESSES AND SIGNALING',
            'P': 'CELLULAR PROCESSES AND SIGNALING',
            'T': 'CELLULAR PROCESSES AND SIGNALING',
            'U': 'CELLULAR PROCESSES AND SIGNALING',
            'V': 'CELLULAR PROCESSES AND SIGNALING',
            'W': 'CELLULAR PROCESSES AND SIGNALING',
            'C': 'METABOLISM',
            'G': 'METABOLISM',
            'E': 'METABOLISM',
            'F': 'METABOLISM',
            'H': 'METABOLISM',
            'I': 'METABOLISM',
            'Q': 'METABOLISM',
            'R': 'POORLY CHARACTERIZED',
            'S': 'POORLY CHARACTERIZED'
        }
        
        logger.info(f"🔬 Benchmark RAG + Ollama inicializado")
        logger.info(f"🔍 RAG: {self.rag_retriever.config['total_entries']:,} entradas")
        logger.info(f"🦙 Ollama timeout: {timeout}s")
    
    def run_benchmark(self, model_name: str, n_samples: int = 100, 
                     use_rag: bool = True, max_rag_evidence: int = 3,
                     generation_params: Dict = None, output_prefix: str = "benchmark") -> Dict:
        """Ejecuta benchmark completo"""
        
        if not self.ollama_client.is_model_available(model_name):
            raise ValueError(f"Modelo {model_name} no disponible. Disponibles: {self.ollama_client.available_models}")
        
        logger.info(f"🚀 INICIANDO BENCHMARK")
        logger.info(f"🤖 Modelo: {model_name}")
        logger.info(f"📊 Muestras: {n_samples}")
        logger.info(f"🔍 RAG: {'✅' if use_rag else '❌'}")
        logger.info(f"📋 Evidencia RAG: {max_rag_evidence if use_rag else 0}")
        
        # Parámetros de generación por defecto
        if generation_params is None:
            generation_params = {
                'temperature': 0.1,
                'top_p': 0.9,
                'top_k': 40
            }
        
        logger.info(f"⚙️ Parámetros: {generation_params}")
        
        # Obtener secuencias de testing
        test_sequences = self.rag_retriever.get_random_test_sequences(n_samples)
        
        # Ejecutar predicciones
        results = []
        total_time = 0
        total_prompt_tokens = 0
        total_response_tokens = 0
        
        successful = 0
        failed = 0
        
        logger.info(f"🔬 Ejecutando predicciones...")
        
        for i, test_seq in enumerate(test_sequences, 1):
            start_time = time.time()
            
            try:
                # Predicción RAG (si está habilitada)
                rag_prediction = None
                if use_rag:
                    rag_prediction = self.rag_retriever.predict_category(
                        test_seq['sequence'], 
                        k=10,
                        exclude_protein_id=test_seq['protein_id']
                    )
                
                # Generar prompt
                evidence = rag_prediction.get('evidence', [])[:max_rag_evidence] if rag_prediction else []
                prompt = self.prompt_generator.generate_enhanced_prompt(
                    test_seq['sequence'], 
                    evidence, 
                    include_evidence=use_rag
                )
                
                # Predicción Ollama
                ollama_result = self.ollama_client.generate_text(
                    model_name, 
                    prompt, 
                    **generation_params
                )
                
                if ollama_result and 'response' in ollama_result:
                    ollama_prediction = self.prompt_generator.extract_prediction_from_response(
                        ollama_result['response']
                    )
                    
                    # Crear resultado del benchmark
                    processing_time = time.time() - start_time
                    
                    result = BenchmarkResult(
                        sequence_id=i,
                        sequence=test_seq['sequence'],
                        true_category=test_seq['category_id'],
                        true_group=self.group_mapping.get(test_seq['category_id'], 'UNKNOWN'),
                        rag_prediction=rag_prediction['predicted_category'] if rag_prediction else 'N/A',
                        rag_confidence=rag_prediction['confidence'] if rag_prediction else 0.0,
                        ollama_prediction=ollama_prediction['predicted_category'],
                        ollama_confidence=ollama_prediction['confidence'],
                        ollama_analysis=ollama_prediction['analysis'],
                        ensemble_prediction=ollama_prediction['predicted_category'],  # Para simplicidad
                        ensemble_confidence=self._calculate_ensemble_confidence(rag_prediction, ollama_prediction),
                        processing_time=processing_time,
                        prompt_tokens=ollama_result.get('prompt_eval_count', 0),
                        response_tokens=ollama_result.get('eval_count', 0),
                        rag_evidence=evidence,
                        raw_ollama_response=ollama_result['response']
                    )
                    
                    results.append(result)
                    successful += 1
                    
                    total_time += processing_time
                    total_prompt_tokens += result.prompt_tokens
                    total_response_tokens += result.response_tokens
                    
                else:
                    failed += 1
                    logger.warning(f"⚠️ Fallo en secuencia {i}")
                
                # Progreso cada 10 secuencias
                if i % 10 == 0:
                    avg_time = total_time / successful if successful > 0 else 0
                    logger.info(f"   Progreso: {i}/{n_samples} | ✅ {successful} | ❌ {failed} | ⏱️ {avg_time:.2f}s/seq")
                    
            except Exception as e:
                failed += 1
                logger.error(f"❌ Error en secuencia {i}: {e}")
                continue
        
        # Calcular métricas
        metrics = self._calculate_metrics(results)
        
        # Agregar estadísticas de rendimiento
        metrics['performance'] = {
            'total_samples': len(results),
            'successful_predictions': successful,
            'failed_predictions': failed,
            'success_rate': successful / n_samples * 100,
            'total_time_seconds': total_time,
            'avg_time_per_prediction': total_time / successful if successful > 0 else 0,
            'total_prompt_tokens': total_prompt_tokens,
            'total_response_tokens': total_response_tokens,
            'avg_prompt_tokens': total_prompt_tokens / successful if successful > 0 else 0,
            'avg_response_tokens': total_response_tokens / successful if successful > 0 else 0,
            'model_name': model_name,
            'generation_params': generation_params,
            'use_rag': use_rag,
            'max_rag_evidence': max_rag_evidence
        }
        
        # Guardar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = self._save_results(results, metrics, f"{output_prefix}_{model_name.replace(':', '_')}_{timestamp}")
        
        logger.info(f"🎉 BENCHMARK COMPLETADO")
        logger.info(f"✅ Exitosas: {successful}/{n_samples} ({successful/n_samples*100:.1f}%)")
        logger.info(f"⏱️ Tiempo total: {total_time:.1f}s")
        logger.info(f"🎯 Precisión categoría: {metrics['category_accuracy']:.3f}")
        logger.info(f"🎯 Precisión grupo: {metrics['group_accuracy']:.3f}")
        logger.info(f"📁 Archivos guardados: {', '.join(output_files)}")
        
        return {
            'results': results,
            'metrics': metrics,
            'output_files': output_files
        }
    
    def _calculate_ensemble_confidence(self, rag_pred: Dict, ollama_pred: Dict) -> float:
        """Calcula confianza ensemble simple"""
        if not rag_pred:
            return 0.5
        
        # Si coinciden las predicciones, alta confianza
        if rag_pred['predicted_category'] == ollama_pred['predicted_category']:
            return min(0.9, 0.5 + 0.4 * rag_pred['confidence'])
        else:
            return 0.6  # Confianza media si no coinciden
    
    def _calculate_metrics(self, results: List[BenchmarkResult]) -> Dict:
        """Calcula métricas de evaluación"""
        if not results:
            return {}
        
        # Métricas básicas
        total = len(results)
        
        # Precisión por categoría
        rag_category_correct = sum(1 for r in results if r.rag_prediction == r.true_category and r.rag_prediction != 'N/A')
        ollama_category_correct = sum(1 for r in results if r.ollama_prediction == r.true_category)
        ensemble_category_correct = sum(1 for r in results if r.ensemble_prediction == r.true_category)
        
        # Precisión por grupo funcional
        ollama_group_correct = sum(1 for r in results if self.group_mapping.get(r.ollama_prediction) == r.true_group)
        ensemble_group_correct = sum(1 for r in results if self.group_mapping.get(r.ensemble_prediction) == r.true_group)
        
        # Contar predicciones RAG válidas
        rag_valid = sum(1 for r in results if r.rag_prediction != 'N/A')
        
        # Métricas de categoría
        metrics = {
            'category_accuracy': ollama_category_correct / total,
            'group_accuracy': ollama_group_correct / total,
            'ensemble_category_accuracy': ensemble_category_correct / total,
            'ensemble_group_accuracy': ensemble_group_correct / total,
        }
        
        # Métricas RAG si está disponible
        if rag_valid > 0:
            metrics['rag_category_accuracy'] = rag_category_correct / rag_valid
            metrics['rag_coverage'] = rag_valid / total
            
            # Acuerdo entre RAG y Ollama
            agreement = sum(1 for r in results if r.rag_prediction == r.ollama_prediction and r.rag_prediction != 'N/A')
            metrics['rag_ollama_agreement'] = agreement / rag_valid
        
        # Métricas por categoría
        category_metrics = defaultdict(lambda: {'total': 0, 'correct': 0, 'precision': 0, 'recall': 0})
        
        # Contar por categoría real
        for result in results:
            true_cat = result.true_category
            pred_cat = result.ollama_prediction
            
            category_metrics[true_cat]['total'] += 1
            if pred_cat == true_cat:
                category_metrics[true_cat]['correct'] += 1
        
        # Calcular precision y recall por categoría
        for category in category_metrics:
            total = category_metrics[category]['total']
            correct = category_metrics[category]['correct']
            
            if total > 0:
                category_metrics[category]['recall'] = correct / total
            
            # Precision: de todas las predicciones de esta categoría, cuántas son correctas
            predicted_as_category = sum(1 for r in results if r.ollama_prediction == category)
            if predicted_as_category > 0:
                category_metrics[category]['precision'] = correct / predicted_as_category
        
        metrics['per_category'] = dict(category_metrics)
        
        # Matriz de confusión simplificada (top categorías)
        from collections import Counter
        true_categories = [r.true_category for r in results]
        pred_categories = [r.ollama_prediction for r in results]
        
        confusion_data = []
        unique_categories = sorted(set(true_categories + pred_categories))
        
        for true_cat in unique_categories:
            for pred_cat in unique_categories:
                count = sum(1 for r in results if r.true_category == true_cat and r.ollama_prediction == pred_cat)
                if count > 0:
                    confusion_data.append({
                        'true_category': true_cat,
                        'predicted_category': pred_cat,
                        'count': count
                    })
        
        metrics['confusion_matrix'] = confusion_data
        
        # Estadísticas de confianza
        confidences = [r.ollama_confidence for r in results]
        high_conf = sum(1 for c in confidences if c == 'High')
        medium_conf = sum(1 for c in confidences if c == 'Medium')
        low_conf = sum(1 for c in confidences if c == 'Low')
        
        metrics['confidence_distribution'] = {
            'high': high_conf / total,
            'medium': medium_conf / total,
            'low': low_conf / total
        }
        
        # Precisión por nivel de confianza
        high_conf_correct = sum(1 for r in results if r.ollama_confidence == 'High' and r.ollama_prediction == r.true_category)
        medium_conf_correct = sum(1 for r in results if r.ollama_confidence == 'Medium' and r.ollama_prediction == r.true_category)
        low_conf_correct = sum(1 for r in results if r.ollama_confidence == 'Low' and r.ollama_prediction == r.true_category)
        
        metrics['accuracy_by_confidence'] = {
            'high': high_conf_correct / high_conf if high_conf > 0 else 0,
            'medium': medium_conf_correct / medium_conf if medium_conf > 0 else 0,
            'low': low_conf_correct / low_conf if low_conf > 0 else 0
        }
        
        return metrics
    
    def _save_results(self, results: List[BenchmarkResult], metrics: Dict, output_prefix: str) -> List[str]:
        """Guarda resultados del benchmark"""
        output_files = []
        
        # 1. CSV con inputs/outputs detallados
        detailed_csv = f"{output_prefix}_detailed.csv"
        detailed_data = []
        
        for result in results:
            # Preparar evidencia RAG como string
            rag_evidence_str = ""
            if result.rag_evidence:
                evidence_parts = []
                for i, evidence in enumerate(result.rag_evidence[:3], 1):
                    evidence_parts.append(
                        f"E{i}: {evidence['similarity']:.3f} | {evidence['category_id']} | {evidence['cog_id']} | {evidence['organism'][:30]}"
                    )
                rag_evidence_str = " || ".join(evidence_parts)
            
            detailed_data.append({
                'sequence_id': result.sequence_id,
                'protein_sequence': result.sequence,
                'sequence_length': len(result.sequence),
                'true_category': result.true_category,
                'true_group': result.true_group,
                'rag_prediction': result.rag_prediction,
                'rag_confidence': result.rag_confidence,
                'ollama_prediction': result.ollama_prediction,
                'ollama_confidence': result.ollama_confidence,
                'ensemble_prediction': result.ensemble_prediction,
                'ensemble_confidence': result.ensemble_confidence,
                'category_correct': result.ollama_prediction == result.true_category,
                'group_correct': self.group_mapping.get(result.ollama_prediction) == result.true_group,
                'processing_time_seconds': result.processing_time,
                'prompt_tokens': result.prompt_tokens,
                'response_tokens': result.response_tokens,
                'ollama_analysis': result.ollama_analysis,
                'rag_evidence': rag_evidence_str,
                'raw_ollama_response': result.raw_ollama_response
            })
        
        pd.DataFrame(detailed_data).to_csv(detailed_csv, index=False)
        output_files.append(detailed_csv)
        
        # 2. CSV con métricas resumen
        metrics_csv = f"{output_prefix}_metrics.csv"
        metrics_data = []
        
        # Métricas generales
        general_metrics = [
            ('category_accuracy', metrics.get('category_accuracy', 0)),
            ('group_accuracy', metrics.get('group_accuracy', 0)),
            ('ensemble_category_accuracy', metrics.get('ensemble_category_accuracy', 0)),
            ('ensemble_group_accuracy', metrics.get('ensemble_group_accuracy', 0)),
        ]
        
        if 'rag_category_accuracy' in metrics:
            general_metrics.extend([
                ('rag_category_accuracy', metrics['rag_category_accuracy']),
                ('rag_coverage', metrics['rag_coverage']),
                ('rag_ollama_agreement', metrics['rag_ollama_agreement'])
            ])
        
        for metric_name, value in general_metrics:
            metrics_data.append({
                'metric_type': 'general',
                'metric_name': metric_name,
                'value': value,
                'category': 'ALL'
            })
        
        # Métricas por categoría
        if 'per_category' in metrics:
            for category, cat_metrics in metrics['per_category'].items():
                for metric_name, value in cat_metrics.items():
                    metrics_data.append({
                        'metric_type': 'per_category',
                        'metric_name': metric_name,
                        'value': value,
                        'category': category
                    })
        
        # Métricas de confianza
        if 'confidence_distribution' in metrics:
            for conf_level, value in metrics['confidence_distribution'].items():
                metrics_data.append({
                    'metric_type': 'confidence_distribution',
                    'metric_name': f'proportion_{conf_level}',
                    'value': value,
                    'category': conf_level.upper()
                })
        
        if 'accuracy_by_confidence' in metrics:
            for conf_level, value in metrics['accuracy_by_confidence'].items():
                metrics_data.append({
                    'metric_type': 'accuracy_by_confidence',
                    'metric_name': f'accuracy_{conf_level}',
                    'value': value,
                    'category': conf_level.upper()
                })
        
        pd.DataFrame(metrics_data).to_csv(metrics_csv, index=False)
        output_files.append(metrics_csv)
        
        # 3. JSON con métricas completas
        metrics_json = f"{output_prefix}_metrics.json"
        with open(metrics_json, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        output_files.append(metrics_json)
        
        # 4. Reporte de texto legible
        report_txt = f"{output_prefix}_report.txt"
        with open(report_txt, 'w') as f:
            f.write("RAG + Ollama COG Benchmark Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Información del experimento
            perf = metrics.get('performance', {})
            f.write(f"Experiment Configuration:\n")
            f.write(f"  Model: {perf.get('model_name', 'Unknown')}\n")
            f.write(f"  Use RAG: {perf.get('use_rag', 'Unknown')}\n")
            f.write(f"  RAG Evidence: {perf.get('max_rag_evidence', 'Unknown')}\n")
            f.write(f"  Generation Params: {perf.get('generation_params', {})}\n")
            f.write(f"  Total Samples: {perf.get('total_samples', 0)}\n")
            f.write(f"  Successful: {perf.get('successful_predictions', 0)}\n")
            f.write(f"  Failed: {perf.get('failed_predictions', 0)}\n")
            f.write(f"  Success Rate: {perf.get('success_rate', 0):.1f}%\n\n")
            
            # Métricas principales
            f.write(f"Main Metrics:\n")
            f.write(f"  Category Accuracy: {metrics.get('category_accuracy', 0):.3f}\n")
            f.write(f"  Group Accuracy: {metrics.get('group_accuracy', 0):.3f}\n")
            
            if 'rag_category_accuracy' in metrics:
                f.write(f"  RAG Category Accuracy: {metrics['rag_category_accuracy']:.3f}\n")
                f.write(f"  RAG Coverage: {metrics['rag_coverage']:.3f}\n")
                f.write(f"  RAG-Ollama Agreement: {metrics['rag_ollama_agreement']:.3f}\n")
            
            f.write(f"\n")
            
            # Rendimiento
            f.write(f"Performance:\n")
            f.write(f"  Total Time: {perf.get('total_time_seconds', 0):.1f} seconds\n")
            f.write(f"  Avg Time/Prediction: {perf.get('avg_time_per_prediction', 0):.2f} seconds\n")
            f.write(f"  Total Prompt Tokens: {perf.get('total_prompt_tokens', 0):,}\n")
            f.write(f"  Total Response Tokens: {perf.get('total_response_tokens', 0):,}\n")
            f.write(f"  Avg Prompt Tokens: {perf.get('avg_prompt_tokens', 0):.1f}\n")
            f.write(f"  Avg Response Tokens: {perf.get('avg_response_tokens', 0):.1f}\n\n")
            
            # Métricas por categoría (top 10)
            if 'per_category' in metrics:
                f.write(f"Per Category Performance (Top 10 by total samples):\n")
                sorted_categories = sorted(
                    metrics['per_category'].items(),
                    key=lambda x: x[1]['total'],
                    reverse=True
                )
                
                for category, cat_metrics in sorted_categories[:10]:
                    f.write(f"  {category}: {cat_metrics['total']} samples, "
                           f"Recall: {cat_metrics['recall']:.3f}, "
                           f"Precision: {cat_metrics['precision']:.3f}\n")
                f.write(f"\n")
            
            # Distribución de confianza
            if 'confidence_distribution' in metrics:
                f.write(f"Confidence Distribution:\n")
                conf_dist = metrics['confidence_distribution']
                f.write(f"  High: {conf_dist['high']:.3f}\n")
                f.write(f"  Medium: {conf_dist['medium']:.3f}\n")
                f.write(f"  Low: {conf_dist['low']:.3f}\n\n")
            
            # Precisión por confianza
            if 'accuracy_by_confidence' in metrics:
                f.write(f"Accuracy by Confidence Level:\n")
                acc_conf = metrics['accuracy_by_confidence']
                f.write(f"  High Confidence: {acc_conf['high']:.3f}\n")
                f.write(f"  Medium Confidence: {acc_conf['medium']:.3f}\n")
                f.write(f"  Low Confidence: {acc_conf['low']:.3f}\n\n")
            
            f.write(f"Files Generated:\n")
            for i, file in enumerate(output_files, 1):
                f.write(f"  {i}. {file}\n")
        
        output_files.append(report_txt)
        
        return output_files

def parse_arguments():
    """Parsea argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='Benchmark RAG + Ollama con secuencias COG aleatorias',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

1. Benchmark básico con modelo específico:
   python rag_ollama_benchmark.py --model llama3.2:3b --samples 100

2. Solo Ollama (sin RAG):
   python rag_ollama_benchmark.py --model mistral:7b --samples 50 --no-rag

3. Benchmark con parámetros personalizados:
   python rag_ollama_benchmark.py --model qwen2.5:7b --samples 200 --temperature 0.05 --top-p 0.85

4. Evaluar múltiples configuraciones:
   python rag_ollama_benchmark.py --model llama3.2:3b --samples 100 --max-evidence 5
   
5. Listar modelos disponibles:
   python rag_ollama_benchmark.py --list-models

ARCHIVOS GENERADOS:
- *_detailed.csv: Inputs/outputs completos con prompts y respuestas
- *_metrics.csv: Métricas estructuradas
- *_metrics.json: Métricas completas en JSON
- *_report.txt: Reporte legible

MÉTRICAS CALCULADAS:
- Precisión por categoría y grupo funcional
- Métricas RAG (si está habilitado)
- Distribución de confianza
- Precisión por nivel de confianza
- Matriz de confusión
- Estadísticas de rendimiento (tiempo, tokens)
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Modelo de Ollama a evaluar (requerido)'
    )
    
    parser.add_argument(
        '--samples', '-s',
        type=int,
        default=100,
        help='Número de secuencias aleatorias a evaluar (default: 100)'
    )
    
    parser.add_argument(
        '--rag-path',
        type=str,
        default='RAG_COG',
        help='Ruta al directorio RAG COG (default: RAG_COG)'
    )
    
    parser.add_argument(
        '--no-rag',
        action='store_true',
        help='Evaluar solo Ollama sin RAG'
    )
    
    parser.add_argument(
        '--max-evidence',
        type=int,
        default=3,
        help='Máximo número de evidencias RAG en prompt (default: 3)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='Temperatura para generación (default: 0.1)'
    )
    
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.9,
        help='Top-p para generación (default: 0.9)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='Top-k para generación (default: 40)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=120,
        help='Timeout por predicción en segundos (default: 120)'
    )
    
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='benchmark',
        help='Prefijo para archivos de salida (default: benchmark)'
    )
    
    parser.add_argument(
        '--ollama-host',
        type=str,
        default='localhost',
        help='Host de Ollama (default: localhost)'
    )
    
    parser.add_argument(
        '--ollama-port',
        type=int,
        default=11434,
        help='Puerto de Ollama (default: 11434)'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='Solo listar modelos disponibles y salir'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mostrar información detallada'
    )
    
    return parser.parse_args()

def main():
    """Función principal"""
    args = parse_arguments()
    
    print(f"🔬 RAG + Ollama COG Benchmark")
    print(f"{'='*50}")
    
    try:
        # Inicializar benchmark
        benchmark = RAGOllamaBenchmark(
            rag_path=args.rag_path,
            ollama_host=args.ollama_host,
            ollama_port=args.ollama_port,
            timeout=args.timeout
        )
        
        # Listar modelos si se solicita
        if args.list_models:
            print(f"\n🦙 MODELOS OLLAMA DISPONIBLES:")
            for model in benchmark.ollama_client.available_models:
                print(f"   • {model}")
            return
        
        # Validar modelo
        if not benchmark.ollama_client.is_model_available(args.model):
            print(f"❌ Modelo '{args.model}' no disponible")
            print(f"📋 Modelos disponibles: {', '.join(benchmark.ollama_client.available_models)}")
            return
        
        # Configurar parámetros de generación
        generation_params = {
            'temperature': args.temperature,
            'top_p': args.top_p,
            'top_k': args.top_k
        }
        
        # Configurar uso de RAG
        use_rag = not args.no_rag
        
        print(f"\n⚙️ CONFIGURACIÓN:")
        print(f"   🤖 Modelo: {args.model}")
        print(f"   📊 Muestras: {args.samples}")
        print(f"   🔍 RAG: {'✅ Habilitado' if use_rag else '❌ Deshabilitado'}")
        if use_rag:
            print(f"   📋 Evidencias RAG: {args.max_evidence}")
        print(f"   🎛️ Parámetros: {generation_params}")
        print(f"   ⏱️ Timeout: {args.timeout}s")
        print(f"   📁 Salida: {args.output_prefix}_*")
        
        # Ejecutar benchmark
        result = benchmark.run_benchmark(
            model_name=args.model,
            n_samples=args.samples,
            use_rag=use_rag,
            max_rag_evidence=args.max_evidence,
            generation_params=generation_params,
            output_prefix=args.output_prefix
        )
        
        # Mostrar resumen final
        metrics = result['metrics']
        
        print(f"\n🎯 RESUMEN FINAL:")
        print(f"   📈 Precisión categoría: {metrics['category_accuracy']:.3f}")
        print(f"   📈 Precisión grupo: {metrics['group_accuracy']:.3f}")
        
        if 'rag_category_accuracy' in metrics:
            print(f"   🔍 Precisión RAG: {metrics['rag_category_accuracy']:.3f}")
            print(f"   🔍 Cobertura RAG: {metrics['rag_coverage']:.3f}")
            print(f"   🤝 Acuerdo RAG-Ollama: {metrics['rag_ollama_agreement']:.3f}")
        
        perf = metrics['performance']
        print(f"   ⏱️ Tiempo promedio: {perf['avg_time_per_prediction']:.2f}s/predicción")
        print(f"   📝 Tokens promedio: {perf['avg_prompt_tokens']:.0f} prompt + {perf['avg_response_tokens']:.0f} respuesta")
        
        print(f"\n📁 ARCHIVOS GENERADOS:")
        for file in result['output_files']:
            print(f"   • {file}")
        
        print(f"\n💡 PRÓXIMOS PASOS:")
        print(f"   📊 Revisar {args.output_prefix}_*_detailed.csv para inputs/outputs completos")
        print(f"   📈 Analizar {args.output_prefix}_*_metrics.csv para métricas detalladas")
        print(f"   📋 Leer {args.output_prefix}_*_report.txt para resumen legible")
        
        # Recomendaciones basadas en resultados
        if metrics['category_accuracy'] < 0.6:
            print(f"\n⚠️ RECOMENDACIONES:")
            print(f"   • Precisión baja ({metrics['category_accuracy']:.3f}), considera:")
            print(f"     - Ajustar temperatura (actual: {args.temperature})")
            print(f"     - Probar otro modelo")
            print(f"     - Aumentar evidencias RAG (actual: {args.max_evidence})")
        
        if use_rag and 'rag_ollama_agreement' in metrics and metrics['rag_ollama_agreement'] < 0.7:
            print(f"   • Bajo acuerdo RAG-Ollama ({metrics['rag_ollama_agreement']:.3f})")
            print(f"     - El modelo podría beneficiarse de fine-tuning")
            print(f"     - Considera ajustar el prompt o evidencias RAG")
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"💡 Asegúrate de que el RAG esté construido en: {args.rag_path}")
        print(f"💡 Ejecuta primero: python rag_cog_builder.py --build")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()