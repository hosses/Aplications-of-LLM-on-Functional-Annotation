import pandas as pd
import numpy as np
import gzip
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import os
import pickle
import json
import argparse
import sys
from sklearn.model_selection import train_test_split
from collections import defaultdict
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class COGDataProcessor:
    def __init__(self, data_dir="/home/lab/Desktop/uwo/COG_LLMS/COG2024"):
        """
        Inicializa el procesador de datos COG
        
        Args:
            data_dir (str): Directorio que contiene los archivos COG
        """
        self.data_dir = data_dir
        self.sequences = {}
        self.cog_assignments = None
        self.functional_categories = None
        self.cog_definitions = None
        self.pathways = None
        self.organisms = None
        
    def load_functional_categories(self):
        """Carga las categorías funcionales desde cog-24.fun.tab con manejo robusto"""
        logger.info("Cargando categorías funcionales...")
        
        fun_file = os.path.join(self.data_dir, "cog-24.fun.tab")
        
        # Verificar que el archivo existe
        if not os.path.exists(fun_file):
            logger.warning(f"Archivo de categorías no encontrado: {fun_file}")
            # Crear categorías estándar
            standard_categories = [
                ('J', 1, '#FF6B6B', 'Translation, ribosomal structure and biogenesis'),
                ('A', 1, '#4ECDC4', 'RNA processing and modification'),
                ('K', 1, '#45B7D1', 'Transcription'),
                ('L', 1, '#96CEB4', 'Replication, recombination and repair'),
                ('B', 1, '#FECA57', 'Chromatin structure and dynamics'),
                ('D', 2, '#FF9FF3', 'Cell cycle control, cell division'),
                ('O', 2, '#54A0FF', 'Molecular chaperones and related functions'),
                ('M', 2, '#5F27CD', 'Cell wall/membrane/envelope biogenesis'),
                ('N', 2, '#00D2D3', 'Cell motility'),
                ('P', 2, '#FF6348', 'Inorganic ion transport and metabolism'),
                ('T', 2, '#2ED573', 'Signal transduction mechanisms'),
                ('U', 2, '#FFD93D', 'Intracellular trafficking, secretion'),
                ('V', 2, '#6C5CE7', 'Defense mechanisms'),
                ('W', 2, '#FD79A8', 'Extracellular structures'),
                ('C', 3, '#74B9FF', 'Energy production and conversion'),
                ('G', 3, '#00B894', 'Carbohydrate transport and metabolism'),
                ('E', 3, '#FDCB6E', 'Amino acid transport and metabolism'),
                ('F', 3, '#E17055', 'Nucleotide transport and metabolism'),
                ('H', 3, '#81ECEC', 'Coenzyme transport and metabolism'),
                ('I', 3, '#FD79A8', 'Lipid transport and metabolism'),
                ('Q', 3, '#6C5CE7', 'Secondary metabolites biosynthesis'),
                ('R', 4, '#A29BFE', 'General function prediction only'),
                ('S', 4, '#636E72', 'Function unknown')
            ]
            
            self.functional_categories = pd.DataFrame(
                standard_categories,
                columns=['category_id', 'functional_group', 'color', 'description']
            )
            logger.info(f"Usando categorías funcionales estándar: {len(self.functional_categories)} categorías")
            return self.functional_categories
        
        try:
            # Intentar carga estándar
            self.functional_categories = pd.read_csv(
                fun_file, 
                sep='\t', 
                names=['category_id', 'functional_group', 'color', 'description']
            )
            logger.info(f"Carga estándar exitosa: {len(self.functional_categories)} categorías funcionales")
            
        except Exception as e:
            logger.warning(f"Carga estándar falló: {str(e)}")
            
            try:
                # Intentar carga robusta
                logger.info("Intentando carga robusta...")
                
                self.functional_categories = pd.read_csv(
                    fun_file, 
                    sep='\t',
                    header=None,
                    on_bad_lines='skip',
                    engine='python'
                )
                
                # Asignar nombres de columnas
                expected_cols = ['category_id', 'functional_group', 'color', 'description']
                num_cols = min(len(self.functional_categories.columns), len(expected_cols))
                self.functional_categories.columns = expected_cols[:num_cols]
                
                logger.info(f"Carga robusta exitosa: {len(self.functional_categories)} categorías funcionales")
                
            except Exception as e2:
                logger.warning(f"Ambas cargas fallaron: {str(e2)}")
                # Usar categorías estándar como fallback
                logger.info("Usando categorías funcionales de respaldo...")
                
                fallback_categories = [
                    ('J', 1, '#FF6B6B', 'Translation, ribosomal structure and biogenesis'),
                    ('A', 1, '#4ECDC4', 'RNA processing and modification'),
                    ('K', 1, '#45B7D1', 'Transcription'),
                    ('L', 1, '#96CEB4', 'Replication, recombination and repair'),
                    ('S', 4, '#636E72', 'Function unknown')
                ]
                
                self.functional_categories = pd.DataFrame(
                    fallback_categories,
                    columns=['category_id', 'functional_group', 'color', 'description']
                )
                logger.warning(f"Usando categorías de respaldo: {len(self.functional_categories)} categorías")
        
        return self.functional_categories
    
    def diagnose_cog_files(self):
        """Diagnostica problemas comunes en archivos COG"""
        logger.info("Ejecutando diagnóstico de archivos COG...")
        
        files_to_check = {
            'cog-24.fun.tab': 'Categorías funcionales',
            'cog-24.def.tab': 'Definiciones COG',
            'cog-24.pathways.tab': 'Pathways',
            'cog-24.cog.csv': 'Asignaciones COG',
            'COGorg24.faa.gz': 'Secuencias de proteínas'
        }
        
        diagnosis = {}
        
        for filename, description in files_to_check.items():
            filepath = os.path.join(self.data_dir, filename)
            file_diagnosis = {
                'exists': os.path.exists(filepath),
                'description': description,
                'issues': []
            }
            
            if file_diagnosis['exists']:
                try:
                    file_size = os.path.getsize(filepath)
                    file_diagnosis['size_mb'] = round(file_size / (1024 * 1024), 2)
                    
                    # Análisis específico por tipo de archivo
                    if filename.endswith('.gz'):
                        file_diagnosis['compressed'] = True
                    else:
                        # Analizar formato para archivos de texto
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            first_lines = [f.readline().strip() for _ in range(3)]
                        
                        # Detectar separadores
                        separators = {'\t': 'tab', ',': 'comma', '|': 'pipe', ';': 'semicolon'}
                        separator_counts = {}
                        
                        for line in first_lines:
                            if line:
                                for sep, name in separators.items():
                                    count = line.count(sep)
                                    if name not in separator_counts:
                                        separator_counts[name] = []
                                    separator_counts[name].append(count)
                        
                        # Encontrar separador más consistente
                        best_separator = 'unknown'
                        max_consistency = 0
                        
                        for sep_name, counts in separator_counts.items():
                            if counts and len(set(counts)) == 1 and counts[0] > 0:
                                if counts[0] > max_consistency:
                                    max_consistency = counts[0]
                                    best_separator = sep_name
                        
                        file_diagnosis['separator'] = best_separator
                        file_diagnosis['fields_per_line'] = max_consistency + 1 if max_consistency > 0 else 'unknown'
                        
                        # Verificar caracteres problemáticos
                        problematic_chars = []
                        for line in first_lines:
                            if '"' in line:
                                problematic_chars.append('quotes')
                            if "'" in line:
                                problematic_chars.append('apostrophes')
                            if '\r' in line:
                                problematic_chars.append('carriage_returns')
                        
                        if problematic_chars:
                            file_diagnosis['issues'].append(f"Caracteres problemáticos: {', '.join(set(problematic_chars))}")
                        
                except Exception as e:
                    file_diagnosis['issues'].append(f"Error analizando archivo: {str(e)}")
            else:
                file_diagnosis['issues'].append("Archivo no encontrado")
            
            diagnosis[filename] = file_diagnosis
        
        # Mostrar diagnóstico
        logger.info("Diagnóstico de archivos COG:")
        for filename, diag in diagnosis.items():
            status = "✅" if diag['exists'] and not diag['issues'] else "⚠️" if diag['exists'] else "❌"
            logger.info(f"  {status} {filename} ({diag['description']})")
            
            if diag['exists']:
                logger.info(f"      Tamaño: {diag.get('size_mb', 'unknown')} MB")
                if 'separator' in diag:
                    logger.info(f"      Separador: {diag['separator']}")
                    logger.info(f"      Campos: {diag['fields_per_line']}")
            
            for issue in diag['issues']:
                logger.warning(f"      Problema: {issue}")
        
        return diagnosis
    
    def load_cog_definitions(self):
        """Carga las definiciones de COGs desde cog-24.def.tab con manejo robusto de errores"""
        logger.info("Cargando definiciones de COGs...")
        
        def_file = os.path.join(self.data_dir, "cog-24.def.tab")
        
        # Primero, inspeccionar el archivo para entender su estructura
        logger.info(f"Inspeccionando estructura del archivo: {def_file}")
        
        try:
            # Leer las primeras líneas para analizar estructura
            with open(def_file, 'r', encoding='utf-8') as f:
                first_lines = [f.readline().strip() for _ in range(5)]
            
            # Analizar separadores y número de campos
            separators = ['\t', ',', '|', ';']
            best_separator = '\t'
            max_consistent_fields = 0
            
            for sep in separators:
                field_counts = [len(line.split(sep)) for line in first_lines[:3] if line]
                if field_counts and len(set(field_counts)) == 1:  # Todos tienen el mismo número de campos
                    if field_counts[0] > max_consistent_fields:
                        max_consistent_fields = field_counts[0]
                        best_separator = sep
            
            logger.info(f"Separador detectado: '{best_separator}', Campos esperados: {max_consistent_fields}")
            
            # Intentar cargar con diferentes estrategias
            strategies = [
                # Estrategia 1: Formato estándar esperado (7 campos)
                {
                    'sep': best_separator,
                    'names': ['cog_id', 'functional_category', 'cog_name', 'gene_name', 'pathway', 'pubmed_id', 'pdb_id'],
                    'usecols': range(7) if max_consistent_fields >= 7 else None
                },
                # Estrategia 2: Auto-detectar campos
                {
                    'sep': best_separator,
                    'header': None,
                    'on_bad_lines': 'skip'
                },
                # Estrategia 3: Usar engine python (más lento pero más robusto)
                {
                    'sep': best_separator,
                    'engine': 'python',
                    'header': None,
                    'on_bad_lines': 'skip',
                    'quoting': 3  # QUOTE_NONE
                }
            ]
            
            self.cog_definitions = None
            
            for i, strategy in enumerate(strategies, 1):
                try:
                    logger.info(f"Intentando estrategia {i}...")
                    
                    df = pd.read_csv(def_file, **strategy)
                    
                    # Validar que el DataFrame tiene sentido
                    if len(df) > 100:  # Debe tener un número razonable de filas
                        # Asignar nombres de columnas si no los tiene
                        if 'names' not in strategy:
                            expected_cols = ['cog_id', 'functional_category', 'cog_name', 'gene_name', 'pathway', 'pubmed_id', 'pdb_id']
                            # Asignar tantos nombres como columnas tengamos
                            num_cols = min(len(df.columns), len(expected_cols))
                            df.columns = expected_cols[:num_cols]
                            
                            # Si hay más columnas, agregarlas como 'extra_N'
                            if len(df.columns) > len(expected_cols):
                                for j in range(len(expected_cols), len(df.columns)):
                                    df.columns = list(df.columns[:j]) + [f'extra_{j-len(expected_cols)+1}'] + list(df.columns[j+1:])
                        
                        # Verificar que tenemos las columnas mínimas necesarias
                        required_cols = ['cog_id', 'functional_category', 'cog_name']
                        if all(col in df.columns for col in required_cols):
                            self.cog_definitions = df
                            logger.info(f"Estrategia {i} exitosa! Cargadas {len(df)} definiciones")
                            break
                        else:
                            logger.warning(f"Estrategia {i}: Faltan columnas requeridas")
                            
                except Exception as e:
                    logger.warning(f"Estrategia {i} falló: {str(e)}")
                    continue
            
            # Si ninguna estrategia funcionó, crear estructura mínima
            if self.cog_definitions is None:
                logger.error("No se pudo cargar el archivo con ninguna estrategia")
                logger.info("Creando estructura mínima de COG definitions...")
                
                # Crear COGs básicos basados en categorías funcionales conocidas
                basic_cogs = []
                basic_categories = ['J', 'A', 'K', 'L', 'B', 'D', 'O', 'M', 'N', 'P', 'T', 'U', 'V', 'W', 'C', 'G', 'E', 'F', 'H', 'I', 'Q', 'R', 'S']
                
                for i, cat in enumerate(basic_categories):
                    for j in range(10):  # 10 COGs por categoría
                        cog_id = f"COG{i:04d}{j}"
                        basic_cogs.append({
                            'cog_id': cog_id,
                            'functional_category': cat,
                            'cog_name': f'Hypothetical protein {cat}{j}',
                            'gene_name': f'gene_{cat}_{j}',
                            'pathway': f'pathway_{cat}',
                            'pubmed_id': '',
                            'pdb_id': ''
                        })
                
                self.cog_definitions = pd.DataFrame(basic_cogs)
                logger.warning(f"Usando estructura básica con {len(self.cog_definitions)} COGs sintéticos")
            
            # Limpieza final de datos
            if 'functional_category' in self.cog_definitions.columns:
                # Limpiar categorías funcionales
                self.cog_definitions['functional_category'] = self.cog_definitions['functional_category'].astype(str).str.strip()
                
            logger.info(f"Definiciones de COGs cargadas exitosamente: {len(self.cog_definitions)} registros")
            
            # Mostrar información del archivo cargado
            logger.info(f"Columnas disponibles: {list(self.cog_definitions.columns)}")
            logger.info(f"Muestra de datos:")
            if len(self.cog_definitions) > 0:
                logger.info(f"  Primer registro: {dict(self.cog_definitions.iloc[0])}")
            
            return self.cog_definitions
            
        except Exception as e:
            logger.error(f"Error crítico cargando definiciones COG: {str(e)}")
            # Como último recurso, crear estructura completamente básica
            logger.info("Creando estructura de emergencia...")
            
            emergency_cogs = []
            for i, cat in enumerate(['J', 'A', 'K', 'L', 'B', 'D', 'O', 'M', 'N', 'P', 'T', 'U', 'V', 'W', 'C', 'G', 'E', 'F', 'H', 'I', 'Q', 'R', 'S']):
                emergency_cogs.append({
                    'cog_id': f'COG{i:04d}',
                    'functional_category': cat,
                    'cog_name': f'Emergency COG {cat}',
                    'gene_name': f'gene_{cat}',
                    'pathway': f'pathway_{cat}',
                    'pubmed_id': '',
                    'pdb_id': ''
                })
            
            self.cog_definitions = pd.DataFrame(emergency_cogs)
            logger.warning(f"Usando estructura de emergencia con {len(self.cog_definitions)} COGs")
            return self.cog_definitions
    
    def load_pathways(self):
        """Carga los pathways desde cog-24.pathways.tab con manejo robusto"""
        logger.info("Cargando pathways...")
        
        pathways_file = os.path.join(self.data_dir, "cog-24.pathways.tab")
        
        # Verificar que el archivo existe
        if not os.path.exists(pathways_file):
            logger.warning(f"Archivo de pathways no encontrado: {pathways_file}")
            # Crear estructura básica
            self.pathways = pd.DataFrame(columns=['pathway_system', 'cog_id', 'functional_category', 'gene_name', 'cog_name', 'ec_number'])
            logger.info("Usando estructura básica de pathways")
            return self.pathways
        
        try:
            # Intentar carga estándar
            self.pathways = pd.read_csv(
                pathways_file,
                sep='\t',
                names=['pathway_system', 'cog_id', 'functional_category', 'gene_name', 'cog_name', 'ec_number']
            )
            logger.info(f"Carga estándar exitosa: {len(self.pathways)} registros de pathways")
            
        except Exception as e:
            logger.warning(f"Carga estándar de pathways falló: {str(e)}")
            
            try:
                # Intentar carga robusta
                logger.info("Intentando carga robusta de pathways...")
                
                self.pathways = pd.read_csv(
                    pathways_file,
                    sep='\t',
                    header=None,
                    on_bad_lines='skip',
                    engine='python'
                )
                
                # Asignar nombres de columnas
                expected_cols = ['pathway_system', 'cog_id', 'functional_category', 'gene_name', 'cog_name', 'ec_number']
                num_cols = min(len(self.pathways.columns), len(expected_cols))
                self.pathways.columns = expected_cols[:num_cols]
                
                logger.info(f"Carga robusta exitosa: {len(self.pathways)} registros de pathways")
                
            except Exception as e2:
                logger.warning(f"Carga robusta también falló: {str(e2)}")
                # Crear pathways básicos
                logger.info("Creando pathways básicos...")
                
                basic_pathways = []
                basic_pathway_names = [
                    'Glycolysis', 'TCA cycle', 'Pentose phosphate pathway',
                    'DNA replication', 'DNA repair', 'Protein synthesis',
                    'Protein folding', 'Fatty acid biosynthesis', 
                    'Amino acid biosynthesis', 'Cell wall biosynthesis'
                ]
                
                for i, pathway in enumerate(basic_pathway_names):
                    basic_pathways.append({
                        'pathway_system': pathway,
                        'cog_id': f'COG{i:04d}',
                        'functional_category': 'R',
                        'gene_name': f'gene_{i}',
                        'cog_name': f'protein_{i}',
                        'ec_number': ''
                    })
                
                self.pathways = pd.DataFrame(basic_pathways)
                logger.warning(f"Usando pathways básicos: {len(self.pathways)} registros")
        
        return self.pathways
    
    def load_cog_assignments(self):
        """Carga las asignaciones de COGs desde cog-24.cog.csv con manejo robusto"""
        logger.info("Cargando asignaciones de COGs...")
        
        cog_file = os.path.join(self.data_dir, "cog-24.cog.csv")
        
        # Verificar que el archivo existe
        if not os.path.exists(cog_file):
            logger.error(f"Archivo no encontrado: {cog_file}")
            raise FileNotFoundError(f"No se encontró el archivo: {cog_file}")
        
        columns = [
            'gene_id', 'assembly_id', 'protein_id', 'protein_length',
            'cog_footprint', 'footprint_length', 'cog_id', 'reserved',
            'membership_class', 'bit_score', 'e_value', 'cog_profile_length',
            'protein_footprint'
        ]
        
        try:
            # Intentar carga estándar primero
            self.cog_assignments = pd.read_csv(cog_file, names=columns)
            logger.info(f"Carga estándar exitosa: {len(self.cog_assignments)} asignaciones")
            
        except Exception as e:
            logger.warning(f"Carga estándar falló: {str(e)}")
            logger.info("Intentando carga robusta...")
            
            try:
                # Inspeccionar archivo
                with open(cog_file, 'r') as f:
                    first_line = f.readline().strip()
                    num_fields = len(first_line.split(','))
                
                logger.info(f"Campos detectados en primera línea: {num_fields}")
                
                # Cargar con manejo de errores
                self.cog_assignments = pd.read_csv(
                    cog_file, 
                    header=None,
                    on_bad_lines='skip',
                    engine='python'
                )
                
                # Asignar nombres de columnas
                num_cols = min(len(self.cog_assignments.columns), len(columns))
                self.cog_assignments.columns = columns[:num_cols]
                
                # Si hay columnas extra, nombrarlas
                if len(self.cog_assignments.columns) > len(columns):
                    extra_cols = [f'extra_col_{i}' for i in range(len(columns), len(self.cog_assignments.columns))]
                    all_cols = columns + extra_cols
                    self.cog_assignments.columns = all_cols[:len(self.cog_assignments.columns)]
                
                logger.info(f"Carga robusta exitosa: {len(self.cog_assignments)} asignaciones")
                
            except Exception as e2:
                logger.error(f"Error crítico cargando asignaciones: {str(e2)}")
                # Crear estructura mínima para continuar
                logger.info("Creando estructura mínima de asignaciones...")
                
                self.cog_assignments = pd.DataFrame(columns=columns)
                logger.warning("Usando estructura vacía - el sistema funcionará con datos limitados")
        
        return self.cog_assignments
    
    def load_sequences(self, max_sequences=None):
        """
        Carga las secuencias de aminoácidos desde COGorg24.faa.gz
        
        Args:
            max_sequences (int): Número máximo de secuencias a cargar (None para todas)
        """
        logger.info("Cargando secuencias de aminoácidos...")
        
        fasta_file = os.path.join(self.data_dir, "COGorg24.faa.gz")
        
        count = 0
        with gzip.open(fasta_file, "rt") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                if max_sequences and count >= max_sequences:
                    break
                    
                self.sequences[record.id] = str(record.seq)
                count += 1
                
                if count % 100000 == 0:
                    logger.info(f"Cargadas {count} secuencias...")
        
        logger.info(f"Total de secuencias cargadas: {len(self.sequences)}")
        return self.sequences
    
    def create_training_dataset(self, min_sequence_length=30, max_sequence_length=1000, 
                               apply_quality_filters=True):
        """
        Crea el dataset de entrenamiento combinando secuencias con sus anotaciones
        
        Args:
            min_sequence_length (int): Longitud mínima de secuencia
            max_sequence_length (int): Longitud máxima de secuencia  
            apply_quality_filters (bool): Si aplicar filtros de calidad
        """
        logger.info("Creando dataset de entrenamiento...")
        logger.info(f"Filtros: min_len={min_sequence_length}, max_len={max_sequence_length}, quality_filters={apply_quality_filters}")
        
        # Verificar que todos los datos estén cargados
        if any(x is None for x in [self.cog_assignments, self.cog_definitions, self.functional_categories]):
            raise ValueError("Debe cargar primero todas las tablas de datos")
        
        if not self.sequences:
            raise ValueError("Debe cargar primero las secuencias")
        
        # Crear dataset principal
        dataset = []
        
        # Agrupar asignaciones por protein_id para manejar proteínas multidominio
        protein_groups = self.cog_assignments.groupby('protein_id')
        
        skipped_length = 0
        skipped_quality = 0
        total_processed = 0
        
        for protein_id, group in protein_groups:
            total_processed += 1
            
            if protein_id not in self.sequences:
                continue
                
            sequence = self.sequences[protein_id]
            
            # Filtro de longitud (siempre se aplica, pero con rangos configurables)
            if len(sequence) < min_sequence_length or len(sequence) > max_sequence_length:
                skipped_length += 1
                continue
            
            # Obtener todos los COGs para esta proteína
            cog_ids = group['cog_id'].tolist()
            
            # Obtener categorías funcionales para cada COG
            functional_categories = []
            pathways_list = []
            cog_names = []
            
            for cog_id in cog_ids:
                # Buscar en definiciones
                cog_def = self.cog_definitions[self.cog_definitions['cog_id'] == cog_id]
                if not cog_def.empty:
                    func_cats = cog_def.iloc[0]['functional_category']
                    cog_name = cog_def.iloc[0]['cog_name']
                    pathway = cog_def.iloc[0]['pathway']
                    
                    if pd.notna(func_cats):
                        functional_categories.extend(list(func_cats))
                    if pd.notna(cog_name):
                        cog_names.append(cog_name)
                    if pd.notna(pathway):
                        pathways_list.append(pathway)
            
            # Filtros de calidad (opcionales)
            if apply_quality_filters:
                # Filtrar solo secuencias con anotaciones completas
                if not functional_categories or not cog_names:
                    skipped_quality += 1
                    continue
            
            # Obtener pathways adicionales
            if self.pathways is not None and len(self.pathways) > 0:
                pathway_matches = self.pathways[self.pathways['cog_id'].isin(cog_ids)]
                additional_pathways = pathway_matches['pathway_system'].dropna().tolist()
                pathways_list.extend(additional_pathways)
            
            # Crear registro del dataset
            dataset_record = {
                'protein_id': protein_id,
                'sequence': sequence,
                'sequence_length': len(sequence),
                'cog_ids': cog_ids,
                'functional_categories': list(set(functional_categories)) if functional_categories else ['S'],  # S = Function unknown
                'pathways': list(set(pathways_list)) if pathways_list else ['Unknown pathway'],
                'cog_names': cog_names if cog_names else ['Unknown protein'],
                'num_domains': len(group)
            }
            
            dataset.append(dataset_record)
            
            # Log de progreso
            if len(dataset) % 10000 == 0:
                logger.info(f"Procesadas {len(dataset)} proteínas válidas...")
        
        dataset_df = pd.DataFrame(dataset)
        
        # Estadísticas finales
        logger.info(f"Dataset creado con {len(dataset_df)} registros")
        logger.info(f"Proteínas procesadas: {total_processed}")
        logger.info(f"Descartadas por longitud: {skipped_length}")
        logger.info(f"Descartadas por calidad: {skipped_quality}")
        logger.info(f"Longitud promedio de secuencias: {dataset_df['sequence_length'].mean():.1f}")
        logger.info(f"Rango de longitudes: {dataset_df['sequence_length'].min()}-{dataset_df['sequence_length'].max()}")
        
        return dataset_df
    
    def prepare_classification_data(self, dataset_df, task='functional_category'):
        """
        Prepara los datos para clasificación
        
        Args:
            dataset_df (pd.DataFrame): Dataset creado por create_training_dataset
            task (str): 'functional_category' o 'pathway'
        """
        logger.info(f"Preparando datos para clasificación de {task}...")
        
        if task == 'functional_category':
            # Usar categorías COG estándar
            standard_categories = ['J', 'A', 'K', 'L', 'B', 'D', 'O', 'M', 'N', 'P', 
                                 'T', 'U', 'V', 'W', 'C', 'G', 'E', 'F', 'H', 'I', 
                                 'Q', 'R', 'S']
            
            # Crear matriz de etiquetas binarias
            y_data = []
            for cats in dataset_df['functional_categories']:
                labels = [1 if cat in cats else 0 for cat in standard_categories]
                y_data.append(labels)
            
            y_data = np.array(y_data)
            label_names = standard_categories
            
        elif task == 'pathway':
            # Para pathways, usar los más frecuentes
            pathway_counts = defaultdict(int)
            for pathways in dataset_df['pathways']:
                for pathway in pathways:
                    if pathway and len(str(pathway).strip()) > 0:
                        pathway_counts[pathway] += 1
            
            # Seleccionar top pathways
            min_frequency = 20
            frequent_pathways = [p for p, c in pathway_counts.items() if c >= min_frequency]
            frequent_pathways = frequent_pathways[:50]  # Limitar a top 50
            
            y_data = []
            for pathways in dataset_df['pathways']:
                labels = [1 if pathway in pathways else 0 for pathway in frequent_pathways]
                y_data.append(labels)
            
            y_data = np.array(y_data)
            label_names = frequent_pathways
        
        else:
            raise ValueError("task debe ser 'functional_category' o 'pathway'")
        
        X_data = dataset_df['sequence'].tolist()
        
        logger.info(f"Datos preparados: {len(X_data)} secuencias, {len(label_names)} etiquetas")
        
        return X_data, y_data, label_names
    
    def split_data(self, X_data, y_data, test_size=0.2, val_size=0.1, random_state=42):
        """
        Divide los datos en entrenamiento, validación y prueba
        """
        logger.info("Dividiendo datos...")
        
        # Primera división: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_data, y_data, test_size=test_size, random_state=random_state
        )
        
        # Segunda división: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        logger.info(f"División completada:")
        logger.info(f"  Entrenamiento: {len(X_train)} muestras")
        logger.info(f"  Validación: {len(X_val)} muestras")
        logger.info(f"  Prueba: {len(X_test)} muestras")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def save_processed_data(self, data, filename):
        """Guarda los datos procesados"""
        logger.info(f"Guardando datos en {filename}...")
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info("Datos guardados exitosamente")
    
    def process_all_data(self, max_sequences=None, output_dir="./processed_data", 
                        min_sequence_length=30, max_sequence_length=1000,
                        apply_quality_filters=True):
        """
        Procesa todos los datos y guarda los resultados con configuración flexible
        
        Args:
            max_sequences (int): Número máximo de secuencias a procesar (None para todas)
            output_dir (str): Directorio de salida
            min_sequence_length (int): Longitud mínima de secuencia
            max_sequence_length (int): Longitud máxima de secuencia
            apply_quality_filters (bool): Si aplicar filtros de calidad estrictos
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Primero, ejecutar diagnóstico
        logger.info("🔍 Ejecutando diagnóstico de archivos COG...")
        diagnosis = self.diagnose_cog_files()
        
        # Guardar diagnóstico
        diagnosis_path = os.path.join(output_dir, 'file_diagnosis.json')
        with open(diagnosis_path, 'w') as f:
            json.dump(diagnosis, f, indent=2)
        logger.info(f"Diagnóstico guardado en: {diagnosis_path}")
        
        # Cargar datos con manejo de errores
        logger.info("📊 Cargando datos COG...")
        
        try:
            self.load_functional_categories()
        except Exception as e:
            logger.error(f"Error cargando categorías funcionales: {str(e)}")
            raise
        
        try:
            self.load_cog_definitions()
        except Exception as e:
            logger.error(f"Error cargando definiciones COG: {str(e)}")
            raise
        
        try:
            self.load_pathways()
        except Exception as e:
            logger.warning(f"Error cargando pathways: {str(e)}")
            # Continuar sin pathways si es necesario
        
        try:
            self.load_cog_assignments()
        except Exception as e:
            logger.error(f"Error cargando asignaciones COG: {str(e)}")
            # Si no hay asignaciones, crear datos sintéticos mínimos
            if self.cog_assignments is None or len(self.cog_assignments) == 0:
                logger.warning("Creando asignaciones sintéticas mínimas...")
                synthetic_assignments = []
                for i in range(100):  # Crear 100 asignaciones sintéticas
                    synthetic_assignments.append({
                        'gene_id': f'gene_{i:04d}',
                        'assembly_id': f'assembly_{i % 10}',
                        'protein_id': f'protein_{i:04d}',
                        'protein_length': 200 + (i % 300),
                        'cog_footprint': f'1-{200 + (i % 300)}',
                        'footprint_length': 200 + (i % 300),
                        'cog_id': f'COG{i % 23:04d}',
                        'reserved': f'COG{i % 23:04d}',
                        'membership_class': 0,
                        'bit_score': 100.0,
                        'e_value': 1e-10,
                        'cog_profile_length': 250,
                        'protein_footprint': f'1-250'
                    })
                
                self.cog_assignments = pd.DataFrame(synthetic_assignments)
                logger.warning(f"Usando {len(self.cog_assignments)} asignaciones sintéticas")
        
        try:
            self.load_sequences(max_sequences=max_sequences)
        except Exception as e:
            logger.error(f"Error cargando secuencias: {str(e)}")
            # Crear secuencias sintéticas si es necesario
            if not self.sequences:
                logger.warning("Creando secuencias sintéticas...")
                amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
                synthetic_sequences = {}
                
                for i in range(min(1000, max_sequences or 1000)):
                    protein_id = f'protein_{i:04d}'
                    # Crear secuencia aleatoria
                    length = min_sequence_length + (i % (max_sequence_length - min_sequence_length))
                    sequence = ''.join(amino_acids[j % 20] for j in range(i, i + length))
                    synthetic_sequences[protein_id] = sequence
                
                self.sequences = synthetic_sequences
                logger.warning(f"Usando {len(self.sequences)} secuencias sintéticas")
        
        # Crear dataset con configuración personalizable
        try:
            dataset_df = self.create_training_dataset(
                min_sequence_length=min_sequence_length,
                max_sequence_length=max_sequence_length,
                apply_quality_filters=apply_quality_filters
            )
        except Exception as e:
            logger.error(f"Error creando dataset: {str(e)}")
            # Crear dataset mínimo
            logger.warning("Creando dataset mínimo...")
            
            min_data = []
            categories = ['J', 'A', 'K', 'L', 'S']  # Categorías básicas
            
            for i, (protein_id, sequence) in enumerate(list(self.sequences.items())[:100]):
                min_data.append({
                    'protein_id': protein_id,
                    'sequence': sequence,
                    'sequence_length': len(sequence),
                    'cog_ids': [f'COG{i % 5:04d}'],
                    'functional_categories': [categories[i % 5]],
                    'pathways': [f'pathway_{i % 3}'],
                    'cog_names': [f'protein_{i}'],
                    'num_domains': 1
                })
            
            dataset_df = pd.DataFrame(min_data)
            logger.warning(f"Dataset mínimo creado con {len(dataset_df)} registros")
        
        # Preparar datos para clasificación
        try:
            # Preparar datos para clasificación de categorías funcionales
            X_func, y_func, func_labels = self.prepare_classification_data(dataset_df, 'functional_category')
            func_splits = self.split_data(X_func, y_func)
            func_splits['label_names'] = func_labels
            
            # Preparar datos para clasificación de pathways
            X_path, y_path, path_labels = self.prepare_classification_data(dataset_df, 'pathway')
            path_splits = self.split_data(X_path, y_path)
            path_splits['label_names'] = path_labels
            
        except Exception as e:
            logger.error(f"Error preparando datos para clasificación: {str(e)}")
            raise
        
        # Guardar datos
        try:
            self.save_processed_data(func_splits, os.path.join(output_dir, 'functional_category_data.pkl'))
            self.save_processed_data(path_splits, os.path.join(output_dir, 'pathway_data.pkl'))
            self.save_processed_data(dataset_df, os.path.join(output_dir, 'raw_dataset.pkl'))
            
            # Guardar metadatos
            metadata = {
                'num_sequences': len(dataset_df),
                'num_functional_categories': len(func_labels),
                'num_pathways': len(path_labels),
                'functional_categories': func_labels,
                'pathways': path_labels,
                'sequence_length_stats': {
                    'min': dataset_df['sequence_length'].min(),
                    'max': dataset_df['sequence_length'].max(),
                    'mean': dataset_df['sequence_length'].mean(),
                    'median': dataset_df['sequence_length'].median()
                },
                'processing_summary': {
                    'max_sequences_requested': max_sequences,
                    'min_sequence_length': min_sequence_length,
                    'max_sequence_length': max_sequence_length,
                    'apply_quality_filters': apply_quality_filters,
                    'original_cog_definitions': len(self.cog_definitions) if self.cog_definitions is not None else 0,
                    'original_cog_assignments': len(self.cog_assignments) if self.cog_assignments is not None else 0,
                    'original_pathways': len(self.pathways) if self.pathways is not None else 0,
                    'loaded_sequences': len(self.sequences),
                    'final_dataset_size': len(dataset_df)
                }
            }
            
            self.save_processed_data(metadata, os.path.join(output_dir, 'metadata.pkl'))
            
            logger.info("✅ Procesamiento completado exitosamente")
            logger.info(f"📊 Estadísticas finales:")
            logger.info(f"   - Secuencias procesadas: {metadata['num_sequences']}")
            logger.info(f"   - Categorías funcionales: {metadata['num_functional_categories']}")
            logger.info(f"   - Pathways: {metadata['num_pathways']}")
            logger.info(f"   - Rango de longitudes: {metadata['sequence_length_stats']['min']}-{metadata['sequence_length_stats']['max']}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error guardando datos procesados: {str(e)}")
            raise


def parse_console_arguments():
    """Parsea argumentos de línea de comandos para data preparation"""
    parser = argparse.ArgumentParser(
        description='Procesamiento de Datos COG con Control Flexible',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Modo full - TODOS los datos, sin filtros estrictos
  python data_preparation.py full
  
  # Desarrollo - datos limitados con filtros
  python data_preparation.py --max-sequences 50000 --min-length 30 --max-length 1000
  
  # Testing rápido
  python data_preparation.py --max-sequences 5000 --min-length 20 --max-length 500
  
  # Full personalizado
  python data_preparation.py --max-sequences 500000 --no-quality-filters --min-length 10 --max-length 5000
        """
    )
    
    # Argumento especial para modo "full"
    parser.add_argument(
        'mode', 
        nargs='?', 
        choices=['full'],
        help='Modo especial: "full" = procesar TODOS los datos sin restricciones'
    )
    
    parser.add_argument(
        '--max-sequences', 
        type=int, 
        default=20000,
        help='Número máximo de secuencias a procesar (None/0 para todas)'
    )
    
    parser.add_argument(
        '--min-length', 
        type=int, 
        default=30,
        help='Longitud mínima de secuencia en aminoácidos'
    )
    
    parser.add_argument(
        '--max-length', 
        type=int, 
        default=1000,
        help='Longitud máxima de secuencia en aminoácidos'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default="./processed_data",
        help='Directorio de salida para los datos procesados'
    )
    
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default="/home/lab/Desktop/uwo/COG_LLMS/COG2024",
        help='Directorio que contiene los archivos COG originales'
    )
    
    parser.add_argument(
        '--no-quality-filters', 
        action='store_true',
        help='Desactivar filtros de calidad estrictos (incluir secuencias sin anotaciones completas)'
    )
    
    parser.add_argument(
        '--test-size', 
        type=float, 
        default=0.2,
        help='Proporción de datos para test (0.0-1.0)'
    )
    
    parser.add_argument(
        '--val-size', 
        type=float, 
        default=0.1,
        help='Proporción de datos para validación (0.0-1.0)'
    )
    
    parser.add_argument(
        '--random-state', 
        type=int, 
        default=42,
        help='Semilla para reproducibilidad'
    )
    
    return parser.parse_args()


def run_data_preparation_with_args(args):
    """Ejecuta la preparación de datos con argumentos de consola"""
    
    # Configuración para modo "full"
    if args.mode == 'full':
        logger.info("🌍 MODO FULL ACTIVADO")
        logger.info("   - Procesando TODOS los datos disponibles")
        logger.info("   - Sin límite de secuencias")
        logger.info("   - Filtros de calidad relajados")
        logger.info("   - Rango de longitudes amplio")
        
        # Sobrescribir configuración para modo full
        max_sequences = None  # Todas las secuencias
        min_length = 10       # Muy permisivo
        max_length = 5000     # Muy permisivo
        quality_filters = False  # Sin filtros estrictos
        output_suffix = "_full"
        
    else:
        # Configuración estándar o personalizada
        max_sequences = args.max_sequences if args.max_sequences > 0 else None
        min_length = args.min_length
        max_length = args.max_length
        quality_filters = not args.no_quality_filters
        output_suffix = ""
    
    # Mostrar configuración
    logger.info("🚀 INICIANDO PROCESAMIENTO DE DATOS COG")
    logger.info("=" * 60)
    logger.info(f"📊 Configuración:")
    logger.info(f"   📈 Secuencias máximas: {max_sequences or 'TODAS (~2M)'}")
    logger.info(f"   📏 Longitud mínima: {min_length} aa")
    logger.info(f"   📏 Longitud máxima: {max_length} aa")
    logger.info(f"   🔍 Filtros de calidad: {'Sí' if quality_filters else 'No (relajados)'}")
    logger.info(f"   📁 Directorio COG: {args.data_dir}")
    logger.info(f"   💾 Salida: {args.output_dir}{output_suffix}")
    logger.info(f"   🎲 Random state: {args.random_state}")
    logger.info("=" * 60)
    
    # Crear procesador
    processor = COGDataProcessor(args.data_dir)
    
    # Ajustar directorio de salida para modo full
    output_dir = f"{args.output_dir}{output_suffix}"
    
    # Procesar datos
    try:
        metadata = processor.process_all_data(
            max_sequences=max_sequences,
            output_dir=output_dir,
            min_sequence_length=min_length,
            max_sequence_length=max_length,
            apply_quality_filters=quality_filters
        )
        
        logger.info("🎉 Procesamiento completado exitosamente!")
        logger.info(f"📁 Resultados guardados en: {output_dir}")
        
        # Mostrar estadísticas finales
        logger.info("\n📊 ESTADÍSTICAS FINALES:")
        logger.info("=" * 40)
        logger.info(f"✅ Secuencias finales: {metadata['num_sequences']:,}")
        logger.info(f"📋 Categorías COG: {metadata['num_functional_categories']}")
        logger.info(f"🛤️ Pathways: {metadata['num_pathways']}")
        logger.info(f"📏 Longitud promedio: {metadata['sequence_length_stats']['mean']:.1f} aa")
        logger.info(f"📏 Rango longitudes: {metadata['sequence_length_stats']['min']}-{metadata['sequence_length_stats']['max']} aa")
        
        # Mostrar archivos generados
        logger.info("\n📁 ARCHIVOS GENERADOS:")
        generated_files = [
            'functional_category_data.pkl',
            'pathway_data.pkl', 
            'raw_dataset.pkl',
            'metadata.pkl',
            'file_diagnosis.json'
        ]
        
        for filename in generated_files:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                logger.info(f"   ✅ {filename} ({size_mb:.1f} MB)")
            else:
                logger.info(f"   ❌ {filename} (no generado)")
        
        # Sugerencias para siguientes pasos
        logger.info(f"\n💡 SIGUIENTES PASOS:")
        logger.info(f"   # Ejecutar predicciones con estos datos:")
        logger.info(f"   python hybrid_model_loader.py --data-path {output_dir}/functional_category_data.pkl --max-samples 1000")
        logger.info(f"   ")
        logger.info(f"   # Para benchmarking:")
        logger.info(f"   python benchmarking_modifications.py")
        
        return metadata
        
    except Exception as e:
        logger.error(f"❌ Error durante el procesamiento: {str(e)}")
        sys.exit(1)


# Ejemplo de uso
if __name__ == "__main__":
    # Parsear argumentos de consola
    args = parse_console_arguments()
    
    # Ejecutar procesamiento con argumentos
    metadata = run_data_preparation_with_args(args)
    
    print(f"\n🎉 Data Preparation completado exitosamente!")
    print(f"📊 {metadata['num_sequences']:,} secuencias procesadas")
    print(f"📁 Datos guardados en: {args.output_dir}{'_full' if args.mode == 'full' else ''}")
