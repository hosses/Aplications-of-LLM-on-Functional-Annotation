import pandas as pd
import gzip
import json
import pickle
import numpy as np
from pathlib import Path
import csv
from Bio import SeqIO
import argparse
import sys
import sqlite3
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re
from collections import defaultdict, Counter
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

@dataclass
class COGEntry:
    """Clase para almacenar información de una entrada COG"""
    protein_id: str
    sequence: str
    functional_region: str  # Secuencia extraída usando footprint
    cog_id: str
    category_id: str
    category_description: str
    functional_group: str
    footprint: str
    footprint_length: int
    membership_class: int
    bit_score: float
    e_value: float
    organism: str
    assembly_id: str
    sequence_length: int
    functional_length: int

class ProteinEmbedder:
    """Clase para generar embeddings de proteínas usando modelos pre-entrenados"""
    
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D", device="cpu"):
        """
        Inicializa el modelo de embeddings de proteínas
        
        Args:
            model_name: Modelo de HuggingFace a usar (ESM-2 por defecto)
            device: Dispositivo para computación (cpu/cuda)
        """
        self.device = device
        self.model_name = model_name
        
        print(f"🧬 Cargando modelo de embeddings: {model_name}")
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
            
            print(f"✅ Modelo cargado. Dimensión de embeddings: {self.embedding_dim}")
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            print("💡 Intentando con modelo más ligero...")
            
            # Fallback a modelo más pequeño
            try:
                self.model_name = "facebook/esm2_t6_8M_UR50D"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.to(device)
                self.model.eval()
                
                with torch.no_grad():
                    test_input = self.tokenizer("M", return_tensors="pt").to(device)
                    test_output = self.model(**test_input)
                    self.embedding_dim = test_output.last_hidden_state.shape[-1]
                    
                print(f"✅ Modelo fallback cargado: {self.model_name}")
                print(f"📏 Dimensión: {self.embedding_dim}")
                
            except Exception as e2:
                print(f"❌ Error crítico cargando modelo: {e2}")
                raise e2
    
    def encode(self, sequences: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Genera embeddings para una lista de secuencias
        
        Args:
            sequences: Lista de secuencias de aminoácidos
            batch_size: Tamaño del batch para procesamiento
            
        Returns:
            Array numpy con embeddings
        """
        embeddings = []
        total_batches = (len(sequences) + batch_size - 1) // batch_size
        
        # Barra de progreso para embeddings
        progress_bar = tqdm(
            total=len(sequences),
            desc="🧮 Generando embeddings",
            unit="seq",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i + batch_size]
                
                # Truncar secuencias muy largas (ESM-2 tiene límite ~1024)
                batch = [seq[:1000] if len(seq) > 1000 else seq for seq in batch]
                
                try:
                    # Tokenizar batch
                    inputs = self.tokenizer(
                        batch, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True,
                        max_length=1022  # Dejar espacio para tokens especiales
                    ).to(self.device)
                    
                    # Generar embeddings
                    outputs = self.model(**inputs)
                    
                    # Usar representación CLS token (posición 0) como embedding de la secuencia
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    
                    embeddings.extend(batch_embeddings)
                    
                    # Actualizar barra de progreso
                    progress_bar.update(len(batch))
                    
                    # Información adicional en la barra
                    current_batch = (i // batch_size) + 1
                    progress_bar.set_postfix({
                        "Batch": f"{current_batch}/{total_batches}",
                        "Embed_dim": self.embedding_dim
                    })
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Error procesando batch {i//batch_size + 1}: {e}")
                    # Usar embeddings cero como fallback
                    batch_embeddings = np.zeros((len(batch), self.embedding_dim))
                    embeddings.extend(batch_embeddings)
                    progress_bar.update(len(batch))
        
        progress_bar.close()
        return np.array(embeddings)

class COGRAGBuilder:
    """Constructor del sistema RAG para COG"""
    
    def __init__(self, base_path: str, output_dir: str = "RAG_COG"):
        self.base_path = Path(base_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'rag_build.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Inicializar embedder
        self.embedder = None
        
        # Datos cargados
        self.cog_definitions = {}
        self.protein_to_cog = {}
        self.organism_info = {}
        self.categories = {}
        self.functional_groups = {}
        
        self.logger.info(f"🚀 Inicializando RAG COG Builder")
        self.logger.info(f"📁 Directorio base: {self.base_path}")
        self.logger.info(f"💾 Directorio salida: {self.output_dir}")
    
    def extract_footprint_sequence(self, sequence: str, footprint: str) -> str:
        """
        Extrae la región funcional de una secuencia usando las coordenadas del footprint
        
        Args:
            sequence: Secuencia completa de aminoácidos
            footprint: Coordenadas del footprint (ej: "45-380" o "1-100=201-300")
            
        Returns:
            Secuencia funcional extraída
        """
        if not footprint or footprint == "":
            return sequence
        
        try:
            functional_sequence = ""
            
            if '=' in footprint:
                # Footprint segmentado (ej: "1-100=201-300")
                segments = footprint.split('=')
                for segment in segments:
                    if '-' in segment:
                        start, end = map(int, segment.split('-'))
                        # Ajustar para indexación Python (1-based a 0-based)
                        functional_sequence += sequence[start-1:end]
            else:
                # Footprint simple (ej: "45-380")
                if '-' in footprint:
                    start, end = map(int, footprint.split('-'))
                    functional_sequence = sequence[start-1:end]
                else:
                    # Si no es un rango válido, usar secuencia completa
                    functional_sequence = sequence
            
            return functional_sequence if functional_sequence else sequence
            
        except Exception as e:
            self.logger.warning(f"Error extrayendo footprint '{footprint}': {e}")
            return sequence
    
    def load_cog_definitions(self) -> None:
        """Carga las definiciones de COGs desde cog-24.def.tab"""
        self.logger.info("📖 Cargando definiciones COG...")
        
        def_file = self.base_path / "cog-24.def.tab"
        if not def_file.exists():
            raise FileNotFoundError(f"No se encontró {def_file}")
        
        with open(def_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        cog_id = parts[0]
                        functional_cat = parts[1]
                        cog_name = parts[2]
                        gene_name = parts[3] if len(parts) > 3 else ""
                        pathway = parts[4] if len(parts) > 4 else ""
                        
                        # Tomar la categoría principal (primera letra)
                        main_category = functional_cat[0] if functional_cat else 'X'
                        
                        self.cog_definitions[cog_id] = {
                            'category': main_category,
                            'name': cog_name,
                            'gene_name': gene_name,
                            'pathway': pathway,
                            'full_categories': functional_cat
                        }
        
        self.logger.info(f"✅ Cargadas {len(self.cog_definitions):,} definiciones COG")
    
    def load_functional_categories(self) -> None:
        """Carga las categorías funcionales"""
        self.logger.info("📖 Definiendo categorías funcionales...")
        
        # Categorías COG estándar
        standard_categories = [
            ('J', 1, 'Translation, ribosomal structure and biogenesis'),
            ('A', 1, 'RNA processing and modification'),
            ('K', 1, 'Transcription'),
            ('L', 1, 'Replication, recombination and repair'),
            ('B', 1, 'Chromatin structure and dynamics'),
            ('D', 2, 'Cell cycle control, cell division'),
            ('O', 2, 'Molecular chaperones and related functions'),
            ('M', 2, 'Cell wall/membrane/envelope biogenesis'),
            ('N', 2, 'Cell motility'),
            ('P', 2, 'Inorganic ion transport and metabolism'),
            ('T', 2, 'Signal transduction mechanisms'),
            ('U', 2, 'Intracellular trafficking, secretion'),
            ('V', 2, 'Defense mechanisms'),
            ('W', 2, 'Extracellular structures'),
            ('C', 3, 'Energy production and conversion'),
            ('G', 3, 'Carbohydrate transport and metabolism'),
            ('E', 3, 'Amino acid transport and metabolism'),
            ('F', 3, 'Nucleotide transport and metabolism'),
            ('H', 3, 'Coenzyme transport and metabolism'),
            ('I', 3, 'Lipid transport and metabolism'),
            ('Q', 3, 'Secondary metabolites biosynthesis'),
            ('R', 4, 'General function prediction only'),
            ('S', 4, 'Function unknown')
        ]
        
        functional_groups = {
            1: 'INFORMATION STORAGE AND PROCESSING',
            2: 'CELLULAR PROCESSES AND SIGNALING',
            3: 'METABOLISM',
            4: 'POORLY CHARACTERIZED'
        }
        
        for cat_id, group_id, description in standard_categories:
            self.categories[cat_id] = description
            self.functional_groups[cat_id] = functional_groups[group_id]
        
        self.logger.info(f"✅ Definidas {len(self.categories)} categorías funcionales")
    
    def load_protein_cog_mappings(self) -> None:
        """Carga las asignaciones proteína -> COG con footprints"""
        self.logger.info("📖 Cargando asignaciones proteína-COG con footprints...")
        
        cog_file = self.base_path / "cog-24.cog.csv"
        if not cog_file.exists():
            raise FileNotFoundError(f"No se encontró {cog_file}")
        
        # Contar líneas totales primero para la barra de progreso
        with open(cog_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        processed = 0
        
        # Barra de progreso para carga de mappings
        with tqdm(
            total=total_lines,
            desc="📖 Cargando mappings",
            unit="líneas",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        ) as pbar:
            
            with open(cog_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    pbar.update(1)
                    
                    if len(row) >= 13:
                        gene_id = row[0]
                        assembly_id = row[1]
                        protein_id = row[2]
                        protein_length = int(row[3]) if row[3].isdigit() else 0
                        footprint = row[4]
                        footprint_length = int(row[5]) if row[5].isdigit() else 0
                        cog_id = row[6]
                        membership_class = int(row[8]) if row[8].isdigit() else 3
                        bit_score = float(row[9]) if row[9] and row[9] != '' else 0.0
                        e_value = float(row[10]) if row[10] and row[10] != '' else 1.0
                        
                        # Almacenar información completa
                        self.protein_to_cog[protein_id] = {
                            'cog_id': cog_id,
                            'footprint': footprint,
                            'footprint_length': footprint_length,
                            'membership_class': membership_class,
                            'bit_score': bit_score,
                            'e_value': e_value,
                            'assembly_id': assembly_id,
                            'protein_length': protein_length
                        }
                        
                        processed += 1
                        
                        # Actualizar información en la barra
                        if processed % 50000 == 0:
                            pbar.set_postfix({
                                "Válidos": f"{processed:,}",
                                "Ratio": f"{processed/pbar.n:.1%}"
                            })
        
        self.logger.info(f"✅ Cargadas {len(self.protein_to_cog):,} asignaciones proteína-COG")
    
    def load_organism_info(self) -> None:
        """Carga información de organismos"""
        self.logger.info("📖 Cargando información de organismos...")
        
        org_file = self.base_path / "cog-24.org.csv"
        if not org_file.exists():
            self.logger.warning(f"⚠️ No se encontró {org_file}, continuando sin info de organismos")
            return
        
        with open(org_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 4:
                    assembly_id = row[0]
                    organism_name = row[1]
                    tax_id = row[2]
                    tax_category = row[3]
                    
                    self.organism_info[assembly_id] = {
                        'name': organism_name,
                        'tax_id': tax_id,
                        'tax_category': tax_category
                    }
        
        self.logger.info(f"✅ Cargada información de {len(self.organism_info):,} organismos")
    
    def process_sequences_and_build_rag(self, max_sequences: Optional[int] = None, 
                                       min_footprint_length: int = 30,
                                       max_membership_class: int = 2) -> None:
        """
        Procesa secuencias, extrae footprints y construye el sistema RAG
        
        Args:
            max_sequences: Máximo número de secuencias a procesar (None = todas)
            min_footprint_length: Longitud mínima del footprint
            max_membership_class: Clase máxima de membresía a incluir (0-3)
        """
        self.logger.info("🧬 Procesando secuencias y construyendo RAG...")
        
        fasta_file = self.base_path / "COGorg24.faa.gz"
        if not fasta_file.exists():
            raise FileNotFoundError(f"No se encontró {fasta_file}")
        
        # Inicializar embedder
        if self.embedder is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"🖥️ Usando dispositivo: {device}")
            self.embedder = ProteinEmbedder(device=device)
        
        # Procesar secuencias
        entries = []
        sequences_for_embedding = []
        functional_sequences = []
        
        processed = 0
        valid = 0
        skipped = 0
        
        self.logger.info(f"📊 Filtros aplicados:")
        self.logger.info(f"   - Longitud mínima footprint: {min_footprint_length}")
        self.logger.info(f"   - Clase máxima membresía: {max_membership_class}")
        if max_sequences:
            self.logger.info(f"   - Máximo secuencias: {max_sequences:,}")
        
        # Barra de progreso principal para procesamiento de secuencias
        sequence_progress = tqdm(
            desc="🧬 Procesando secuencias",
            unit="seq",
            bar_format="{l_bar}{bar}| {n_fmt} procesadas [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        try:
            with gzip.open(fasta_file, 'rt') as f:
                for record in SeqIO.parse(f, "fasta"):
                    processed += 1
                    sequence_progress.update(1)
                    
                    protein_id = record.id
                    sequence = str(record.seq)
                    
                    # Verificar si tenemos información COG para esta proteína
                    if protein_id in self.protein_to_cog:
                        cog_mapping = self.protein_to_cog[protein_id]
                        cog_id = cog_mapping['cog_id']
                        
                        # Verificar si tenemos definición del COG
                        if cog_id in self.cog_definitions:
                            cog_info = self.cog_definitions[cog_id]
                            category_id = cog_info['category']
                            
                            # Aplicar filtros de calidad
                            if (cog_mapping['membership_class'] <= max_membership_class and
                                cog_mapping['footprint_length'] >= min_footprint_length and
                                category_id in self.categories):
                                
                                # Extraer región funcional usando footprint
                                functional_region = self.extract_footprint_sequence(
                                    sequence, cog_mapping['footprint']
                                )
                                
                                # Verificar que la región funcional es válida
                                if len(functional_region) >= min_footprint_length:
                                    # Obtener información del organismo
                                    assembly_id = cog_mapping['assembly_id']
                                    organism_name = "Unknown"
                                    if assembly_id in self.organism_info:
                                        organism_name = self.organism_info[assembly_id]['name']
                                    
                                    # Crear entrada COG
                                    entry = COGEntry(
                                        protein_id=protein_id,
                                        sequence=sequence,
                                        functional_region=functional_region,
                                        cog_id=cog_id,
                                        category_id=category_id,
                                        category_description=self.categories[category_id],
                                        functional_group=self.functional_groups[category_id],
                                        footprint=cog_mapping['footprint'],
                                        footprint_length=cog_mapping['footprint_length'],
                                        membership_class=cog_mapping['membership_class'],
                                        bit_score=cog_mapping['bit_score'],
                                        e_value=cog_mapping['e_value'],
                                        organism=organism_name,
                                        assembly_id=assembly_id,
                                        sequence_length=len(sequence),
                                        functional_length=len(functional_region)
                                    )
                                    
                                    entries.append(entry)
                                    functional_sequences.append(functional_region)
                                    valid += 1
                                    
                                    # Actualizar información en la barra
                                    sequence_progress.set_postfix({
                                        "✅": valid,
                                        "❌": skipped,
                                        "Ratio": f"{valid/(valid+skipped):.1%}" if (valid+skipped) > 0 else "0%"
                                    })
                                    
                                    # Verificar límite
                                    if max_sequences and valid >= max_sequences:
                                        sequence_progress.set_description("🎯 Límite alcanzado")
                                        break
                                else:
                                    skipped += 1
                            else:
                                skipped += 1
                        else:
                            skipped += 1
                    else:
                        skipped += 1
                        
        except Exception as e:
            sequence_progress.close()
            self.logger.error(f"Error procesando secuencias: {e}")
            raise e
        
        sequence_progress.close()
        
        self.logger.info(f"✅ Procesamiento completado:")
        self.logger.info(f"   📖 Secuencias procesadas: {processed:,}")
        self.logger.info(f"   ✅ Secuencias válidas: {valid:,}")
        self.logger.info(f"   ⏭️ Secuencias omitidas: {skipped:,}")
        
        if not entries:
            raise ValueError("No se encontraron secuencias válidas")
        
        # Generar embeddings
        self.logger.info("🧮 Generando embeddings de regiones funcionales...")
        embeddings = self.embedder.encode(functional_sequences, batch_size=32)
        
        self.logger.info(f"✅ Generados {len(embeddings):,} embeddings de dimensión {embeddings.shape[1]}")
        
        # Construir índice FAISS
        self.logger.info("🔍 Construyendo índice FAISS...")
        dimension = embeddings.shape[1]
        
        # Usar índice IVF para mejor rendimiento con datasets grandes
        nlist = min(int(np.sqrt(len(embeddings))), 1000)  # Número de clusters
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        # Entrenar índice con barra de progreso
        with tqdm(total=1, desc="🏋️ Entrenando índice FAISS", bar_format="{desc}: {bar}") as pbar:
            index.train(embeddings.astype(np.float32))
            pbar.update(1)
        
        # Agregar vectores al índice con barra de progreso
        with tqdm(total=1, desc="📥 Agregando vectores", bar_format="{desc}: {bar}") as pbar:
            index.add(embeddings.astype(np.float32))
            pbar.update(1)
        
        self.logger.info(f"✅ Índice FAISS construido con {index.ntotal:,} vectores")
        
        # Guardar todos los componentes
        self.save_rag_components(entries, embeddings, index)
        
        # Generar estadísticas
        self.generate_statistics(entries)
    
    def save_rag_components(self, entries: List[COGEntry], embeddings: np.ndarray, 
                           faiss_index) -> None:
        """Guarda todos los componentes del RAG"""
        self.logger.info("💾 Guardando componentes del RAG...")
        
        # 1. Guardar metadatos de secuencias en SQLite
        db_path = self.output_dir / "cog_rag.db"
        conn = sqlite3.connect(db_path)
        
        # Crear tabla
        conn.execute('''
            CREATE TABLE IF NOT EXISTS cog_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                protein_id TEXT,
                sequence TEXT,
                functional_region TEXT,
                cog_id TEXT,
                category_id TEXT,
                category_description TEXT,
                functional_group TEXT,
                footprint TEXT,
                footprint_length INTEGER,
                membership_class INTEGER,
                bit_score REAL,
                e_value REAL,
                organism TEXT,
                assembly_id TEXT,
                sequence_length INTEGER,
                functional_length INTEGER
            )
        ''')
        
        # Insertar datos
        for i, entry in enumerate(entries):
            conn.execute('''
                INSERT INTO cog_entries VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            ''', (
                i, entry.protein_id, entry.sequence, entry.functional_region,
                entry.cog_id, entry.category_id, entry.category_description,
                entry.functional_group, entry.footprint, entry.footprint_length,
                entry.membership_class, entry.bit_score, entry.e_value,
                entry.organism, entry.assembly_id, entry.sequence_length,
                entry.functional_length
            ))
        
        conn.commit()
        conn.close()
        
        # 2. Guardar índice FAISS
        faiss_path = str(self.output_dir / "cog_rag.index")
        faiss.write_index(faiss_index, faiss_path)
        
        # 3. Guardar embeddings como backup
        embeddings_path = self.output_dir / "embeddings.npy"
        np.save(embeddings_path, embeddings)
        
        # 4. Guardar configuración y metadatos
        config = {
            'embedding_dim': embeddings.shape[1],
            'total_entries': len(entries),
            'model_name': self.embedder.model_name,
            'build_date': datetime.now().isoformat(),
            'categories': self.categories,
            'functional_groups': self.functional_groups
        }
        
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # 5. Guardar mapeo de IDs para referencia rápida
        id_mapping = {i: entry.protein_id for i, entry in enumerate(entries)}
        mapping_path = self.output_dir / "id_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(id_mapping, f)
        
        self.logger.info(f"✅ RAG guardado en: {self.output_dir}")
        self.logger.info(f"   📊 Base de datos: {db_path}")
        self.logger.info(f"   🔍 Índice FAISS: {faiss_path}")
        self.logger.info(f"   🧮 Embeddings: {embeddings_path}")
        self.logger.info(f"   ⚙️ Configuración: {config_path}")
    
    def generate_statistics(self, entries: List[COGEntry]) -> None:
        """Genera estadísticas del dataset RAG"""
        self.logger.info("📊 Generando estadísticas...")
        
        # Estadísticas básicas
        total_entries = len(entries)
        categories = [entry.category_id for entry in entries]
        functional_groups = [entry.functional_group for entry in entries]
        organisms = [entry.organism for entry in entries]
        
        # Contadores
        category_counts = Counter(categories)
        group_counts = Counter(functional_groups)
        organism_counts = Counter(organisms)
        
        # Estadísticas de longitud
        seq_lengths = [entry.sequence_length for entry in entries]
        func_lengths = [entry.functional_length for entry in entries]
        
        # Crear reporte
        stats = {
            'total_entries': total_entries,
            'unique_categories': len(category_counts),
            'unique_organisms': len(organism_counts),
            'avg_sequence_length': np.mean(seq_lengths),
            'avg_functional_length': np.mean(func_lengths),
            'category_distribution': dict(category_counts),
            'group_distribution': dict(group_counts),
            'top_organisms': dict(organism_counts.most_common(10)),
            'length_stats': {
                'sequence': {
                    'min': min(seq_lengths),
                    'max': max(seq_lengths),
                    'mean': np.mean(seq_lengths),
                    'std': np.std(seq_lengths)
                },
                'functional': {
                    'min': min(func_lengths),
                    'max': max(func_lengths),
                    'mean': np.mean(func_lengths),
                    'std': np.std(func_lengths)
                }
            }
        }
        
        # Guardar estadísticas
        stats_path = self.output_dir / "statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        # Crear reporte legible
        report_path = self.output_dir / "statistics_report.txt"
        with open(report_path, 'w') as f:
            f.write("COG RAG Statistics Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Build Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Entries: {total_entries:,}\n")
            f.write(f"Unique Categories: {len(category_counts)}\n")
            f.write(f"Unique Organisms: {len(organism_counts)}\n\n")
            
            f.write("Category Distribution:\n")
            f.write("-" * 30 + "\n")
            for cat_id, count in category_counts.most_common():
                desc = self.categories.get(cat_id, 'Unknown')
                pct = (count / total_entries) * 100
                f.write(f"{cat_id}: {desc[:40]}... ({count:,}, {pct:.1f}%)\n")
            
            f.write(f"\nFunctional Group Distribution:\n")
            f.write("-" * 35 + "\n")
            for group, count in group_counts.most_common():
                pct = (count / total_entries) * 100
                f.write(f"{group}: {count:,} ({pct:.1f}%)\n")
            
            f.write(f"\nTop 10 Organisms:\n")
            f.write("-" * 20 + "\n")
            for org, count in organism_counts.most_common(10):
                pct = (count / total_entries) * 100
                f.write(f"{org[:50]}... : {count:,} ({pct:.1f}%)\n")
            
            f.write(f"\nSequence Length Statistics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Full sequences - Mean: {np.mean(seq_lengths):.1f}, "
                   f"Min: {min(seq_lengths)}, Max: {max(seq_lengths)}\n")
            f.write(f"Functional regions - Mean: {np.mean(func_lengths):.1f}, "
                   f"Min: {min(func_lengths)}, Max: {max(func_lengths)}\n")
        
        self.logger.info(f"✅ Estadísticas guardadas en: {stats_path}")
        self.logger.info(f"📄 Reporte legible en: {report_path}")

class COGRAGRetriever:
    """Sistema de recuperación para el RAG COG"""
    
    def __init__(self, rag_path: str = "RAG_COG"):
        self.rag_path = Path(rag_path)
        
        if not self.rag_path.exists():
            raise FileNotFoundError(f"No se encontró el directorio RAG: {rag_path}")
        
        print(f"🔍 Cargando RAG COG desde: {self.rag_path}")
        
        # Cargar configuración
        config_path = self.rag_path / "config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        print(f"✅ Configuración cargada: {self.config['total_entries']:,} entradas")
        
        # Cargar índice FAISS
        faiss_path = str(self.rag_path / "cog_rag.index")
        self.index = faiss.read_index(faiss_path)
        
        print(f"🔍 Índice FAISS cargado: {self.index.ntotal:,} vectores")
        
        # Conectar a base de datos
        db_path = self.rag_path / "cog_rag.db"
        self.conn = sqlite3.connect(db_path)
        
        # Cargar mapeo de IDs
        mapping_path = self.rag_path / "id_mapping.json"
        with open(mapping_path, 'r') as f:
            self.id_mapping = {int(k): v for k, v in json.load(f).items()}
        
        # Inicializar embedder con el mismo modelo
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = ProteinEmbedder(
            model_name=self.config['model_name'], 
            device=device
        )
        
        print(f"🧬 Embedder inicializado: {self.config['model_name']}")
        print(f"✅ RAG COG listo para consultas")
    
    def retrieve(self, query_sequence: str, k: int = 10, 
                min_similarity: float = 0.3) -> List[Dict]:
        """
        Recupera las k secuencias más similares a la consulta
        
        Args:
            query_sequence: Secuencia de proteína a consultar
            k: Número de resultados a devolver
            min_similarity: Similitud mínima (0-1)
            
        Returns:
            Lista de diccionarios con información de las secuencias similares
        """
        print(f"🔍 Buscando secuencias similares a: {query_sequence[:50]}...")
        
        # Generar embedding de la consulta
        query_embedding = self.embedder.encode([query_sequence])
        
        # Buscar en el índice
        distances, indices = self.index.search(
            query_embedding.astype(np.float32), k * 2  # Buscar más para filtrar
        )
        
        # Convertir distancias a similitudes (coseno aproximado)
        similarities = 1 / (1 + distances[0])  # Normalización simple
        
        # Recuperar información de la base de datos
        results = []
        for i, (idx, similarity) in enumerate(zip(indices[0], similarities)):
            if similarity >= min_similarity:
                # Consultar base de datos
                cursor = self.conn.execute(
                    "SELECT * FROM cog_entries WHERE id = ?", (int(idx),)
                )
                row = cursor.fetchone()
                
                if row:
                    result = {
                        'similarity': float(similarity),
                        'protein_id': row[1],
                        'sequence': row[2],
                        'functional_region': row[3],
                        'cog_id': row[4],
                        'category_id': row[5],
                        'category_description': row[6],
                        'functional_group': row[7],
                        'footprint': row[8],
                        'footprint_length': row[9],
                        'membership_class': row[10],
                        'bit_score': row[11],
                        'e_value': row[12],
                        'organism': row[13],
                        'assembly_id': row[14],
                        'sequence_length': row[15],
                        'functional_length': row[16]
                    }
                    results.append(result)
                    
                    if len(results) >= k:
                        break
        
        print(f"✅ Encontradas {len(results)} secuencias similares")
        return results
    
    def predict_category(self, query_sequence: str, k: int = 20) -> Dict:
        """
        Predice la categoría COG de una secuencia usando RAG
        
        Args:
            query_sequence: Secuencia a clasificar
            k: Número de vecinos a considerar
            
        Returns:
            Diccionario con predicción y confianza
        """
        # Recuperar secuencias similares
        similar_sequences = self.retrieve(query_sequence, k=k)
        
        if not similar_sequences:
            return {
                'predicted_category': 'S',  # Function unknown
                'confidence': 0.0,
                'evidence': [],
                'category_scores': {}
            }
        
        # Votar por categoría ponderado por similitud
        category_votes = defaultdict(float)
        total_weight = 0
        
        evidence = []
        for seq_info in similar_sequences:
            weight = seq_info['similarity']
            category = seq_info['category_id']
            
            category_votes[category] += weight
            total_weight += weight
            
            evidence.append({
                'protein_id': seq_info['protein_id'],
                'similarity': seq_info['similarity'],
                'category': category,
                'cog_id': seq_info['cog_id'],
                'organism': seq_info['organism']
            })
        
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
            'category_description': self.config['categories'].get(
                predicted_category, 'Unknown'
            ),
            'functional_group': self.config['functional_groups'].get(
                predicted_category, 'Unknown'
            ),
            'confidence': float(confidence),
            'evidence': evidence[:5],  # Top 5 evidencias
            'category_scores': {k: float(v) for k, v in category_scores.items()}
        }
    
    def get_statistics(self) -> Dict:
        """Obtiene estadísticas del RAG"""
        cursor = self.conn.execute("""
            SELECT 
                category_id,
                COUNT(*) as count,
                AVG(sequence_length) as avg_seq_len,
                AVG(functional_length) as avg_func_len
            FROM cog_entries 
            GROUP BY category_id
            ORDER BY count DESC
        """)
        
        category_stats = {}
        for row in cursor.fetchall():
            category_stats[row[0]] = {
                'count': row[1],
                'avg_sequence_length': row[2],
                'avg_functional_length': row[3],
                'description': self.config['categories'].get(row[0], 'Unknown')
            }
        
        return {
            'total_entries': self.config['total_entries'],
            'embedding_dimension': self.config['embedding_dim'],
            'model_name': self.config['model_name'],
            'categories': category_stats
        }

def main():
    parser = argparse.ArgumentParser(
        description="Constructor de RAG para COG con footprints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

1. Construir RAG completo (LENTO - todas las secuencias):
   python rag_cog_builder.py /ruta/a/cog_databases --build

2. Construir RAG limitado (RÁPIDO - muestra balanceada):
   python rag_cog_builder.py /ruta/a/cog_databases --build --max-sequences 50000

3. Probar consulta en RAG existente:
   python rag_cog_builder.py /ruta/a/cog_databases --query "MLSNKYLKDFIF..."

4. Obtener estadísticas del RAG:
   python rag_cog_builder.py /ruta/a/cog_databases --stats

OPTIMIZACIONES:
- Usa solo regiones funcionales (footprints) para embeddings
- Filtra por calidad (membership class ≤ 2)
- Longitud mínima de footprint: 30 aminoácidos
- Índice FAISS IVF para búsquedas rápidas
- Base de datos SQLite para metadatos

SALIDA:
Carpeta RAG_COG/ con:
- cog_rag.db: Base de datos SQLite con metadatos
- cog_rag.index: Índice FAISS para búsquedas
- embeddings.npy: Embeddings de respaldo
- config.json: Configuración del sistema
- statistics.json: Estadísticas del dataset
        """
    )
    
    parser.add_argument(
        'base_path',
        help='Ruta al directorio que contiene los archivos COG'
    )
    
    parser.add_argument(
        '--build', '-b',
        action='store_true',
        help='Construir el sistema RAG'
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Secuencia de proteína para consultar al RAG'
    )
    
    parser.add_argument(
        '--stats', '-s',
        action='store_true',
        help='Mostrar estadísticas del RAG existente'
    )
    
    parser.add_argument(
        '--max-sequences',
        type=int,
        help='Máximo número de secuencias a procesar (para pruebas rápidas)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='RAG_COG',
        help='Directorio donde guardar el RAG (por defecto: RAG_COG)'
    )
    
    parser.add_argument(
        '--min-footprint',
        type=int,
        default=30,
        help='Longitud mínima del footprint (por defecto: 30)'
    )
    
    parser.add_argument(
        '--max-membership',
        type=int,
        default=2,
        help='Clase máxima de membresía COG (0-3, por defecto: 2)'
    )
    
    args = parser.parse_args()
    
    # Validar directorio
    if not Path(args.base_path).exists():
        print(f"❌ Error: El directorio {args.base_path} no existe")
        sys.exit(1)
    
    try:
        if args.build:
            print("🚀 CONSTRUYENDO RAG COG")
            print("=" * 50)
            
            # Construir RAG
            builder = COGRAGBuilder(args.base_path, args.output_dir)
            builder.load_functional_categories()
            builder.load_cog_definitions()
            builder.load_protein_cog_mappings()
            builder.load_organism_info()
            
            builder.process_sequences_and_build_rag(
                max_sequences=args.max_sequences,
                min_footprint_length=args.min_footprint,
                max_membership_class=args.max_membership
            )
            
            print(f"\n🎉 RAG COG construido exitosamente en: {args.output_dir}")
            
        elif args.query:
            print("🔍 CONSULTANDO RAG COG")
            print("=" * 50)
            
            # Cargar y consultar RAG
            retriever = COGRAGRetriever(args.output_dir)
            
            # Hacer predicción
            prediction = retriever.predict_category(args.query, k=20)
            
            print(f"\n🧬 Secuencia consultada: {args.query[:50]}...")
            print(f"🏷️ Categoría predicha: {prediction['predicted_category']}")
            print(f"📝 Descripción: {prediction['category_description']}")
            print(f"🔬 Grupo funcional: {prediction['functional_group']}")
            print(f"🎯 Confianza: {prediction['confidence']:.3f}")
            
            print(f"\n📊 Top categorías candidatas:")
            for cat, score in sorted(prediction['category_scores'].items(), 
                                   key=lambda x: x[1], reverse=True)[:5]:
                print(f"   {cat}: {score:.3f}")
            
            print(f"\n🔍 Evidencia (Top 5 secuencias similares):")
            for i, evidence in enumerate(prediction['evidence'], 1):
                print(f"   {i}. Similitud: {evidence['similarity']:.3f} | "
                      f"Categoría: {evidence['category']} | "
                      f"COG: {evidence['cog_id']} | "
                      f"Organismo: {evidence['organism'][:30]}...")
                      
        elif args.stats:
            print("📊 ESTADÍSTICAS RAG COG")
            print("=" * 50)
            
            # Cargar y mostrar estadísticas
            retriever = COGRAGRetriever(args.output_dir)
            stats = retriever.get_statistics()
            
            print(f"📈 Total de entradas: {stats['total_entries']:,}")
            print(f"🧮 Dimensión embeddings: {stats['embedding_dimension']}")
            print(f"🤖 Modelo: {stats['model_name']}")
            print(f"🏷️ Categorías: {len(stats['categories'])}")
            
            print(f"\n📊 Top 10 categorías por frecuencia:")
            sorted_cats = sorted(stats['categories'].items(), 
                               key=lambda x: x[1]['count'], reverse=True)
            for cat_id, info in sorted_cats[:10]:
                print(f"   {cat_id}: {info['description'][:40]}... "
                      f"({info['count']:,} secuencias)")
        
        else:
            print("❌ Debe especificar --build, --query, o --stats")
            parser.print_help()
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()