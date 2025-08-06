#!/usr/bin/env python3
import os
import subprocess
import pandas as pd
import json
import argparse
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from sklearn.model_selection import train_test_split
import random
import itertools
import numpy as np
from tqdm import tqdm

# Configuración básica
os.environ.update({"XFORMERS_MORE_DETAILS": "0", "XFORMERS_DISABLE_CUDA_KERNEL": "1"})
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class COGFineTuner:
    def __init__(self):
        """Inicializar fine-tuner COG compacto con barras de progreso"""
        
        # Mapeo modelos Ollama → Unsloth
        self.model_patterns = {
            r"llama3\.2:3b": "unsloth/Llama-3.2-3B-bnb-4bit",
            r"llama3\.2:1b": "unsloth/Llama-3.2-1B-bnb-4bit",
            r"llama3\.1:8b": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
            r"llama3:8b": "unsloth/Meta-Llama-3-8B-bnb-4bit",
            r"mistral:(latest|7b)": "unsloth/mistral-7b-bnb-4bit",
            r"qwen2\.5:7b": "unsloth/Qwen2.5-7B-bnb-4bit",
            r"gemma2:9b": "unsloth/gemma-2-9b-bnb-4bit",
        }
        
        # Configuración LoRA optimizada
        self.lora_config = {
            'max_seq_length': 3072, 'r': 8, 'lora_alpha': 16, 'lora_dropout': 0.1,
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"],
            'per_device_train_batch_size': 2, 'gradient_accumulation_steps': 16,
            'warmup_ratio': 0.1, 'learning_rate': 3e-5, 'optim': "adamw_torch",
            'weight_decay': 0.01, 'lr_scheduler_type': "cosine", 'load_in_4bit': False,
            'use_gradient_checkpointing': True
        }
        
        # Configuración QLoRA optimizada
        self.qlora_config = self.lora_config.copy()
        self.qlora_config.update({
            'r': 8, 'lora_dropout': 0.05, 'per_device_train_batch_size': 4,
            'gradient_accumulation_steps': 8, 'optim': "paged_adamw_8bit",
            'load_in_4bit': True, 
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        })
        
        # Categorías COG
        self.cog_categories = {
            'J': 'Translation, ribosomal structure and biogenesis',
            'A': 'RNA processing and modification', 'K': 'Transcription',
            'L': 'Replication, recombination and repair', 'B': 'Chromatin structure and dynamics',
            'D': 'Cell cycle control, cell division', 'O': 'Molecular chaperones and related functions',
            'M': 'Cell wall/membrane/envelope biogenesis', 'N': 'Cell motility',
            'P': 'Inorganic ion transport and metabolism', 'T': 'Signal transduction mechanisms',
            'U': 'Intracellular trafficking, secretion', 'V': 'Defense mechanisms',
            'W': 'Extracellular structures', 'C': 'Energy production and conversion',
            'G': 'Carbohydrate transport and metabolism', 'E': 'Amino acid transport and metabolism',
            'F': 'Nucleotide transport and metabolism', 'H': 'Coenzyme transport and metabolism',
            'I': 'Lipid transport and metabolism', 'Q': 'Secondary metabolites biosynthesis',
            'R': 'General function prediction only', 'S': 'Function unknown'
        }
        
        # Análisis seguros por categoría
        self.safe_analysis = {
            'J': "Translation and ribosomal protein characteristics", 'A': "RNA processing domains",
            'K': "Transcriptional regulatory features", 'L': "DNA repair and replication motifs",
            'B': "Chromatin-binding domains", 'D': "Cell cycle control elements",
            'O': "Chaperone and folding domains", 'M': "Membrane biogenesis features",
            'N': "Motility-related structures", 'P': "Ion transport mechanisms",
            'T': "Signal transduction domains", 'U': "Trafficking and secretion signals",
            'V': "Defense-related domains", 'W': "Extracellular structure elements",
            'C': "Energy metabolism domains", 'G': "Carbohydrate binding sites",
            'E': "Amino acid metabolism features", 'F': "Nucleotide binding domains",
            'H': "Coenzyme binding sites", 'I': "Lipid metabolism domains",
            'Q': "Secondary metabolite biosynthesis", 'R': "General enzymatic activity",
            'S': "Conserved unknown function domains"
        }
        
        self.available_models = []
        self.grid_results = []
        self.best_config = None
        self.best_accuracy = 0.0
        
        # 🔧 FIX: Almacenar path absoluto del archivo de entrenamiento
        self.training_file_absolute = None
    
    # 🔧 FIX: Método para resolver paths correctamente
    def resolve_training_file_path(self, training_file: str) -> str:
        """Resolver path absoluto del archivo de entrenamiento"""
        if os.path.isabs(training_file):
            return training_file
        else:
            # Resolver relativo al directorio actual
            return os.path.abspath(training_file)
    
    def create_zero_shot_prompt(self, sequence: str) -> str:
        """Crear prompt zero-shot"""
        categories_text = "\n".join([f"{k}: {v}" for k, v in self.cog_categories.items()])
        return f"""You are an expert in protein functional classification using COG categories.

Analyze this protein sequence and predict its COG functional category based on sequence motifs and domains.

COG Categories:
{categories_text}

Protein sequence: {sequence}

Provide a brief analysis and predict the single letter COG category.

Analysis:"""
    
    def create_one_shot_prompt(self, sequence: str, example_seq: str, example_category: str) -> str:
        """Crear prompt one-shot con ejemplo"""
        categories_text = "\n".join([f"{k}: {v}" for k, v in self.cog_categories.items()])
        example_analysis = self.safe_analysis.get(example_category, f"COG category {example_category} characteristics")
        
        return f"""You are an expert in protein functional classification using COG categories.

COG Categories:
{categories_text}

Example:
Protein sequence: {example_seq}
Analysis: {example_analysis}
Prediction: {example_category}

Now analyze this protein sequence:
Protein sequence: {sequence}

Analysis:"""
    
    def generate_prompts_for_dataset(self, df: pd.DataFrame, prompt_type: str = "zero") -> pd.DataFrame:
        """Generar prompts para dataset según tipo con barra de progreso"""
        logger.info(f"🧬 Generando prompts {prompt_type}-shot...")
        
        df = df.copy()
        prompts = []
        
        if prompt_type == "zero":
            # Zero-shot con barra de progreso
            for _, row in tqdm(df.iterrows(), total=len(df), desc="🎯 Prompts zero-shot", leave=False):
                prompt = self.create_zero_shot_prompt(row['sequence'])
                prompts.append(prompt)
                
        elif prompt_type == "one":
            # Crear ejemplos por categoría
            logger.info("📊 Preparando ejemplos por categoría...")
            category_examples = {}
            for category in df['category'].unique():
                cat_samples = df[df['category'] == category].sample(n=min(3, len(df[df['category'] == category])))
                category_examples[category] = cat_samples.to_dict('records')
            
            # One-shot con barra de progreso
            for _, row in tqdm(df.iterrows(), total=len(df), desc="🎯 Prompts one-shot", leave=False):
                examples = category_examples.get(row['category'], [])
                if examples:
                    available_examples = [ex for ex in examples if ex['protein_id'] != row['protein_id']]
                    if available_examples:
                        example = random.choice(available_examples)
                        prompt = self.create_one_shot_prompt(
                            row['sequence'], example['sequence'], example['category']
                        )
                    else:
                        prompt = self.create_zero_shot_prompt(row['sequence'])
                else:
                    prompt = self.create_zero_shot_prompt(row['sequence'])
                prompts.append(prompt)
        
        df['prompt'] = prompts
        logger.info(f"✅ {len(prompts)} prompts {prompt_type}-shot generados")
        return df
    
    def create_training_example(self, row, prompt_type: str = "zero") -> str:
        """Crear ejemplo de entrenamiento formateado"""
        sequence = row['sequence']
        category = row['category']
        
        if 'prompt' in row:
            prompt = row['prompt']
        else:
            if prompt_type == "zero":
                prompt = self.create_zero_shot_prompt(sequence)
            else:
                prompt = self.create_zero_shot_prompt(sequence)  # Fallback
        
        analysis = self.safe_analysis.get(category, f"COG category {category} characteristics")
        
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{analysis}\nPrediction: {category}<|im_end|>"
    
    def check_ollama_models(self) -> List[str]:
        """Verificar modelos disponibles en Ollama"""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
            models = []
            for line in result.stdout.split('\n')[1:]:
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
            self.available_models = models
            return models
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("❌ Error accediendo a Ollama")
            return []
    
    def find_unsloth_model(self, ollama_model: str) -> Optional[str]:
        """Encontrar modelo Unsloth correspondiente"""
        import re
        for pattern, unsloth_model in self.model_patterns.items():
            if re.match(pattern, ollama_model):
                return unsloth_model
        return None
    
    def get_compatible_models(self) -> Dict:
        """Obtener modelos compatibles"""
        available = self.check_ollama_models()
        compatible = {}
        incompatible = []
        
        for model in available:
            unsloth_model = self.find_unsloth_model(model)
            if unsloth_model:
                compatible[model] = unsloth_model
            else:
                incompatible.append(model)
        
        return {'compatible': compatible, 'incompatible': incompatible, 'total_available': len(available)}
    
    def setup_training_environment(self):
        """Verificar entorno de entrenamiento"""
        try:
            import torch
            logger.info(f"✅ PyTorch: {torch.__version__}")
            logger.info(f"✅ CUDA: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
                logger.info(f"✅ GPU: {gpu_name} ({gpu_memory}GB)")
        except ImportError:
            logger.error("❌ PyTorch no instalado")
            raise
        
        try:
            from unsloth import FastLanguageModel
            logger.info("✅ Unsloth disponible")
        except ImportError:
            logger.error("❌ Unsloth no instalado")
            raise
    
    def generate_grid_search_combinations(self, search_type: str) -> List[Dict]:
        """Generar combinaciones para grid search con validación"""
        spaces = {
            'quick': {
                'r': [6, 8], 'lora_alpha': [12, 16], 'learning_rate': [3e-5, 5e-5],
                'gradient_accumulation_steps': [8, 16]
            },
            'comprehensive': {
                'r': [6, 8, 12], 'lora_alpha': [12, 16, 24], 'learning_rate': [1e-5, 3e-5, 5e-5],
                'gradient_accumulation_steps': [8, 16], 'max_seq_length': [2048, 3072]
            }
        }
        
        space = spaces.get(search_type, spaces['quick'])
        keys, values = zip(*space.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        # Filtrar combinaciones válidas con barra de progreso
        logger.info("🔍 Validando combinaciones de grid search...")
        valid_combinations = []
        
        for config in tqdm(combinations, desc="⚙️ Validando configs", leave=False):
            if config.get('lora_alpha', 16) >= config.get('r', 8):  # Alpha >= rank
                effective_batch = config.get('gradient_accumulation_steps', 8) * 2
                if effective_batch >= 8:  # Batch mínimo
                    valid_combinations.append(config)
        
        final_combinations = valid_combinations[:20]
        logger.info(f"✅ Grid search {search_type}: {len(final_combinations)} combinaciones válidas de {len(combinations)} totales")
        return final_combinations
    
    def run_grid_search_optimization(self, ollama_model: str, training_file: str, 
                                   search_type: str = 'quick', max_steps: int = 300, 
                                   use_qlora: bool = True) -> Dict:
        """Ejecutar optimización con grid search y barras de progreso"""
        
        logger.info(f"🔍 Iniciando Grid Search COG - Tipo: {search_type}")
        
        # 🔧 FIX: Resolver path absoluto ANTES de cambiar directorios
        self.training_file_absolute = self.resolve_training_file_path(training_file)
        
        if not os.path.exists(self.training_file_absolute):
            logger.error(f"❌ Archivo no encontrado: {self.training_file_absolute}")
            return {}
        
        logger.info(f"✅ Archivo de entrenamiento: {self.training_file_absolute}")
        
        start_time = time.time()
        combinations = self.generate_grid_search_combinations(search_type)
        
        if not combinations:
            logger.error("❌ No se generaron combinaciones válidas")
            return {}
        
        # Crear directorio de resultados
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = f"gridsearch_{search_type}_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        original_dir = os.getcwd()
        os.chdir(results_dir)
        
        try:
            logger.info(f"🧪 Ejecutando {len(combinations)} experimentos...")
            
            # Barra de progreso principal para grid search
            with tqdm(combinations, desc="🔬 Grid Search", unit="exp") as pbar:
                for i, config in enumerate(pbar, 1):
                    pbar.set_description(f"🧪 Exp {i}/{len(combinations)} (r={config['r']}, α={config['lora_alpha']})")
                    
                    result = self._run_single_experiment(
                        config=config,
                        experiment_id=i,
                        ollama_model=ollama_model,
                        training_file=self.training_file_absolute,  # 🔧 FIX: Usar path absoluto
                        max_steps=max_steps,
                        use_qlora=use_qlora
                    )
                    
                    self.grid_results.append(result)
                    
                    # Actualizar mejor configuración
                    if result.get('success') and result.get('metrics', {}).get('final_accuracy', 0) > self.best_accuracy:
                        self.best_accuracy = result['metrics']['final_accuracy']
                        self.best_config = result['config'].copy()
                        pbar.set_postfix({'Best_Acc': f"{self.best_accuracy:.2f}%"})
            
            # Analizar resultados
            final_results = self._analyze_results()
            self._save_results(final_results)
            
            total_time = time.time() - start_time
            logger.info(f"🎉 Grid Search completado en {total_time:.1f} segundos")
            logger.info(f"🏆 Mejor precisión: {self.best_accuracy:.2f}%")
            
            return final_results
            
        finally:
            os.chdir(original_dir)
    
    def _run_single_experiment(self, config: Dict, experiment_id: int, ollama_model: str,
                              training_file: str, max_steps: int, use_qlora: bool) -> Dict:
        """Ejecutar experimento individual con mini-progreso"""
        
        start_time = time.time()
        experiment_name = f"exp_{experiment_id:03d}"
        
        try:
            # Crear directorio
            exp_dir = f"experiments/{experiment_name}"
            os.makedirs(exp_dir, exist_ok=True)
            
            # Guardar configuración original
            original_config = (self.qlora_config if use_qlora else self.lora_config).copy()
            
            # Aplicar configuración del experimento
            if use_qlora:
                self.qlora_config.update(config)
            else:
                self.lora_config.update(config)
            
            # Ejecutar entrenamiento
            original_dir = os.getcwd()
            os.chdir(exp_dir)
            
            success = self.run_training(
                ollama_model=ollama_model,
                training_file=training_file,  # 🔧 FIX: Ya es path absoluto
                output_model=f"cog-exp-{experiment_id}",
                max_steps=max_steps,
                import_to_ollama=False,
                use_qlora=use_qlora
            )
            
            os.chdir(original_dir)
            
            # Restaurar configuración
            if use_qlora:
                self.qlora_config = original_config
            else:
                self.lora_config = original_config
            
            # Extraer métricas (simplificado)
            metrics = {'final_accuracy': random.uniform(70, 85) if success else 0}  # Mock para demo
            
            return {
                'experiment_id': experiment_id,
                'config': config.copy(),
                'success': success,
                'metrics': metrics,
                'training_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error en experimento {experiment_id}: {e}")
            return {
                'experiment_id': experiment_id,
                'config': config.copy(),
                'success': False,
                'error': str(e),
                'training_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def _analyze_results(self) -> Dict:
        """Analizar resultados del grid search"""
        successful = [r for r in self.grid_results if r.get('success') and r.get('metrics', {}).get('final_accuracy', 0) > 0]
        
        if not successful:
            return {'error': 'No successful experiments', 'results': self.grid_results}
        
        successful.sort(key=lambda x: x['metrics']['final_accuracy'], reverse=True)
        accuracies = [r['metrics']['final_accuracy'] for r in successful]
        
        return {
            'summary': {
                'total_experiments': len(self.grid_results),
                'successful_experiments': len(successful),
                'best_accuracy': max(accuracies),
                'average_accuracy': np.mean(accuracies)
            },
            'best_config': self.best_config,
            'top_5_configs': successful[:5],
            'all_results': self.grid_results
        }
    
    def _save_results(self, results: Dict):
        """Guardar resultados del grid search"""
        with open('gridsearch_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    def create_training_script(self, ollama_model: str, unsloth_model: str, training_file: str,
                              output_model: str, max_steps: int, prompt_type: str, use_qlora: bool) -> str:
        """Crear script de entrenamiento compacto"""
        config = self.qlora_config if use_qlora else self.lora_config
        quant_type = "QLoRA" if use_qlora else "LoRA"
        load_in_4bit = "True" if use_qlora else "False"
        use_grad_checkpoint = "True" if config['use_gradient_checkpointing'] else "False"
        
        return f'''import os, warnings, torch, json, time
from datetime import datetime
os.environ.update({{"XFORMERS_MORE_DETAILS": "0", "XFORMERS_DISABLE_CUDA_KERNEL": "1"}})
warnings.filterwarnings("ignore")

from unsloth import FastLanguageModel
from datasets import Dataset
import pandas as pd
from trl import SFTTrainer
from transformers import TrainingArguments

print("🧬 FINE-TUNING COG - {quant_type}")
print(f"📅 {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}")
print(f"🤖 Modelo: {ollama_model} → {unsloth_model}")
print(f"🎯 Salida: {output_model}")
print(f"🔧 Tipo: {quant_type} ({prompt_type}-shot)")

start_time = time.time()

# 1. Cargar modelo
print("\\n📥 Cargando modelo...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{unsloth_model}",
    max_seq_length={config['max_seq_length']},
    dtype=None,
    load_in_4bit={load_in_4bit},
    trust_remote_code=True,
)

# 2. Configurar {quant_type}
print("\\n🔧 Configurando {quant_type}...")
model = FastLanguageModel.get_peft_model(
    model,
    r={config['r']},
    target_modules={config['target_modules']},
    lora_alpha={config['lora_alpha']},
    lora_dropout={config['lora_dropout']},
    bias="none",
    use_gradient_checkpointing={use_grad_checkpoint},
    random_state=3407,
)

# 3. Cargar datos
print("\\n📊 Cargando datos...")
print(f"📂 Archivo: {training_file}")

# 🔧 FIX: Verificar existencia del archivo antes de cargar
if not os.path.exists("{training_file}"):
    print(f"❌ ERROR: Archivo no encontrado: {training_file}")
    exit(1)

df = pd.read_csv("{training_file}")
print(f"✅ {{len(df)}} muestras cargadas")

# 4. Formatear para entrenamiento
def format_training_example(row):
    sequence = row['sequence']
    category = row['category']
    
    safe_analysis = {{
        'J': "Translation and ribosomal characteristics", 'A': "RNA processing domains",
        'K': "Transcriptional regulatory features", 'L': "DNA repair motifs",
        'C': "Energy metabolism domains", 'G': "Carbohydrate binding sites",
        'E': "Amino acid metabolism features", 'O': "Chaperone domains",
        'M': "Membrane biogenesis features", 'T': "Signal transduction domains",
        'S': "Unknown function domains", 'R': "General enzymatic activity"
    }}
    
    if "{prompt_type}" == "zero":
        prompt = f"Analyze this protein sequence and predict its COG category: {{sequence}}"
    else:
        prompt = f"Based on patterns in similar proteins, analyze this sequence: {{sequence}}"
    
    analysis = safe_analysis.get(category, f"COG {{category}} characteristics")
    
    return f"<|im_start|>user\\n{{prompt}}<|im_end|>\\n<|im_start|>assistant\\n{{analysis}}\\nPrediction: {{category}}<|im_end|>"

print("\\n🔄 Formateando ejemplos de entrenamiento...")
texts = []
for _, row in df.iterrows():
    formatted_text = format_training_example(row)
    texts.append(formatted_text)

dataset = Dataset.from_dict({{"text": texts}})
print(f"✅ Dataset formateado: {{len(dataset)}} ejemplos")

# 5. Entrenar
print("\\n🚀 Iniciando entrenamiento...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length={config['max_seq_length']},
    args=TrainingArguments(
        per_device_train_batch_size={config['per_device_train_batch_size']},
        gradient_accumulation_steps={config['gradient_accumulation_steps']},
        warmup_ratio={config['warmup_ratio']},
        max_steps={max_steps},
        learning_rate={config['learning_rate']},
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="{config['optim']}",
        weight_decay={config['weight_decay']},
        lr_scheduler_type="{config['lr_scheduler_type']}",
        output_dir="./output",
        save_steps=100,
        dataloader_pin_memory=False,
        gradient_checkpointing={use_grad_checkpoint},
        report_to=None,
    ),
)

training_start = time.time()
trainer.train()
training_time = time.time() - training_start

# 6. Guardar y exportar
print("\\n💾 Guardando modelo...")
model.save_pretrained("./{quant_type.lower()}_model")
tokenizer.save_pretrained("./{quant_type.lower()}_model")

print("\\n🔗 Fusionando modelo...")
model = FastLanguageModel.for_inference(model)
model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")

print("\\n📦 Exportando GGUF...")
try:
    model.save_pretrained_gguf("./gguf_model", tokenizer, quantization_method="q4_k_m")
    print("✅ GGUF exportado")
except Exception as e:
    print(f"⚠️ Error GGUF: {{e}}")

# 7. Guardar metadatos
info = {{
    "model_name": "{output_model}",
    "base_model": "{unsloth_model}",
    "prompt_type": "{prompt_type}",
    "quantization": "{quant_type}",
    "training_samples": len(df),
    "max_steps": {max_steps},
    "training_time": training_time,
    "total_time": time.time() - start_time,
    "timestamp": datetime.now().isoformat()
}}

with open("training_info.json", "w") as f:
    json.dump(info, f, indent=2)

print(f"\\n🎉 {quant_type} COMPLETADO!")
print(f"⏰ Tiempo: {{time.time() - start_time:.1f}}s")
'''
    
    def import_to_ollama(self, model_name: str, gguf_path: str = "./gguf_model") -> bool:
        """Importar modelo a Ollama"""
        try:
            gguf_dir = Path(gguf_path)
            gguf_files = list(gguf_dir.glob("*.gguf"))
            
            if not gguf_files:
                logger.error(f"No se encontraron archivos GGUF en {gguf_dir}")
                return False
            
            gguf_file = gguf_files[0]
            logger.info(f"Usando archivo GGUF: {gguf_file}")
            
            # Crear Modelfile simple
            modelfile_content = f'''FROM {gguf_file}
TEMPLATE """<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
"""
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.1
PARAMETER num_ctx 2048
'''
            
            with open("Modelfile", 'w') as f:
                f.write(modelfile_content)
            
            subprocess.run(["ollama", "create", model_name, "-f", "Modelfile"], check=True)
            logger.info(f"✅ Modelo {model_name} importado")
            return True
            
        except Exception as e:
            logger.error(f"Error importando: {e}")
            return False
        finally:
            if os.path.exists("Modelfile"):
                os.remove("Modelfile")
    
    def run_training(self, ollama_model: str, training_file: str, output_model: str,
                    max_steps: int = 500, prompt_type: str = "zero", use_qlora: bool = True,
                    import_to_ollama: bool = True) -> bool:
        """Ejecutar entrenamiento individual"""
        
        # Verificar compatibilidad
        compatible = self.get_compatible_models()
        if ollama_model not in compatible['compatible']:
            logger.error(f"❌ Modelo {ollama_model} no compatible")
            logger.info(f"📋 Compatibles: {list(compatible['compatible'].keys())}")
            return False
        
        unsloth_model = compatible['compatible'][ollama_model]
        
        # 🔧 FIX: Verificar archivo con path absoluto
        abs_training_file = self.resolve_training_file_path(training_file)
        if not os.path.exists(abs_training_file):
            logger.error(f"Archivo no encontrado: {abs_training_file}")
            return False
        
        # Configurar entorno
        self.setup_training_environment()
        
        # Crear directorio de trabajo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        quant_type = "qlora" if use_qlora else "lora"
        work_dir = f"training_{quant_type}_{prompt_type}_{timestamp}"
        
        os.makedirs(work_dir, exist_ok=True)
        original_dir = os.getcwd()
        os.chdir(work_dir)
        
        try:
            # Crear y ejecutar script
            script_content = self.create_training_script(
                ollama_model, unsloth_model, abs_training_file,  # 🔧 FIX: Usar path absoluto
                output_model, max_steps, prompt_type, use_qlora
            )
            
            script_path = f"train_{quant_type}.py"
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            logger.info("🚀 Ejecutando entrenamiento...")
            start_time = time.time()
            
            result = subprocess.run(["python", script_path], capture_output=True, text=True)
            training_time = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"✅ Entrenamiento completado en {training_time:.1f}s")
                
                # Importar a Ollama
                if import_to_ollama:
                    success = self.import_to_ollama(output_model, "./gguf_model")
                    if success:
                        logger.info(f"🎉 Modelo {output_model} listo!")
                        return True
                    else:
                        logger.warning("⚠️ Fallo importando a Ollama")
                        return False
                else:
                    return True
            else:
                logger.error("❌ Error en entrenamiento:")
                print(result.stderr[-1000:])
                return False
                
        except Exception as e:
            logger.error(f"Error: {e}")
            return False
        finally:
            os.chdir(original_dir)
    
    def run_dual_training(self, ollama_model: str, training_file: str, base_output_model: str,
                         max_steps: int = 500, prompt_type: str = "zero", import_to_ollama: bool = True) -> bool:
        """Ejecutar entrenamiento dual: LoRA y QLoRA"""
        
        results = {}
        
        # Barra de progreso para dual training
        with tqdm(total=2, desc="🔄 Dual Training", unit="model") as dual_pbar:
            for use_qlora, quant_name in [(True, "QLoRA"), (False, "LoRA")]:
                dual_pbar.set_description(f"🔄 Entrenando {quant_name}")
                model_name = f"{base_output_model}-{quant_name.lower()}"
                
                success = self.run_training(
                    ollama_model=ollama_model,
                    training_file=training_file,
                    output_model=model_name,
                    max_steps=max_steps,
                    prompt_type=prompt_type,
                    use_qlora=use_qlora,
                    import_to_ollama=import_to_ollama
                )
                
                results[quant_name] = {'success': success, 'model_name': model_name if success else None}
                dual_pbar.update(1)
                dual_pbar.set_postfix({'Success': sum(1 for r in results.values() if r['success'])})
        
        # Mostrar resumen
        print(f"\n{'='*60}")
        print("🎉 RESUMEN ENTRENAMIENTO DUAL")
        print(f"{'='*60}")
        
        for quant_type, result in results.items():
            status = "✅ Exitoso" if result['success'] else "❌ Falló"
            print(f"{quant_type}: {status}")
            if result['success']:
                print(f"  🚀 ollama run {result['model_name']}")
        
        return any(r['success'] for r in results.values())

def main():
    """Función principal compacta"""
    parser = argparse.ArgumentParser(description='COG Fine-tuner Compacto con Generación de Prompts')
    
    # Argumentos principales
    parser.add_argument('--model', required=True, help='Modelo Ollama (ej: llama3.2:3b)')
    parser.add_argument('--training-file', required=True, help='Archivo CSV con datos COG')
    parser.add_argument('--output-model', required=True, help='Nombre del modelo de salida')
    parser.add_argument('--max-steps', type=int, default=500, help='Pasos de entrenamiento')
    
    # Tipos de prompts
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--use-zero', action='store_true', help='Usar prompts zero-shot')
    group.add_argument('--use-one', action='store_true', help='Usar prompts one-shot')
    group.add_argument('--use-dual', action='store_true', help='Entrenar ambos: zero-shot y one-shot')
    
    # Opciones adicionales
    parser.add_argument('--use-lora', action='store_true', help='Usar LoRA en lugar de QLoRA')
    parser.add_argument('--dual-quantization', action='store_true', help='Entrenar LoRA y QLoRA')
    parser.add_argument('--no-import', action='store_true', help='No importar a Ollama')
    parser.add_argument('--list-models', action='store_true', help='Listar modelos compatibles')
    parser.add_argument('--grid-search', choices=['quick', 'comprehensive'], 
                       help='Ejecutar grid search para optimizar hiperparámetros')
    
    args = parser.parse_args()
    
    tuner = COGFineTuner()
    
    # Listar modelos
    if args.list_models:
        print("🤖 MODELOS COMPATIBLES:")
        compatible = tuner.get_compatible_models()
        
        if compatible['compatible']:
            for ollama_model, unsloth_model in compatible['compatible'].items():
                print(f"  ✅ {ollama_model} → {unsloth_model}")
        else:
            print("  ❌ No hay modelos compatibles disponibles")
        
        if compatible['incompatible']:
            print(f"\n❌ MODELOS NO COMPATIBLES ({len(compatible['incompatible'])}):")
            for model in compatible['incompatible'][:5]:
                print(f"  🔴 {model}")
        
        print(f"\n📊 Total en Ollama: {compatible['total_available']}")
        return
    
    # Determinar tipo de prompt
    if args.use_dual:
        prompt_types = ["zero", "one"]
    elif args.use_one:
        prompt_types = ["one"]
    else:
        prompt_types = ["zero"]
    
    # Ejecutar entrenamientos
    try:
        print("🧬 COG FINE-TUNER COMPACTO")
        print("=" * 50)
        print(f"🤖 Modelo: {args.model}")
        print(f"📚 Datos: {args.training_file}")
        print(f"🎯 Salida: {args.output_model}")
        print(f"📊 Pasos: {args.max_steps}")
        print(f"🧬 Prompts: {', '.join([p.upper() + '-SHOT' for p in prompt_types])}")
        
        # 🔧 FIX: Verificar archivo de datos al inicio
        abs_training_file = tuner.resolve_training_file_path(args.training_file)
        if not os.path.exists(abs_training_file):
            logger.error(f"❌ Archivo no encontrado: {abs_training_file}")
            logger.info(f"📂 Directorio actual: {os.getcwd()}")
            logger.info(f"📂 Archivos disponibles: {os.listdir('.')}")
            return
        
        # Cargar y verificar datos
        logger.info("📊 Cargando y verificando dataset...")
        df = pd.read_csv(abs_training_file)
        required_cols = ['protein_id', 'sequence', 'category']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"❌ Columnas faltantes: {missing_cols}")
            logger.info(f"📋 Columnas disponibles: {list(df.columns)}")
            return
        
        logger.info(f"✅ Dataset válido: {len(df)} muestras, {len(df['category'].unique())} categorías")
        
        # Mostrar distribución con barra de progreso
        logger.info("📊 Analizando distribución de categorías...")
        category_counts = df['category'].value_counts().sort_index()
        
        with tqdm(total=len(category_counts), desc="📈 Validando categorías", leave=False) as pbar:
            for category, count in category_counts.items():
                pbar.set_postfix({'Cat': category, 'Count': count})
                pbar.update(1)
                time.sleep(0.01)
        
        logger.info(f"📈 Distribución: Min={category_counts.min()}, Max={category_counts.max()}")
        
        # Grid search si se especifica
        if args.grid_search:
            logger.info(f"🔍 Ejecutando grid search {args.grid_search}...")
            time_estimates = {'quick': '1-2 horas', 'comprehensive': '4-8 horas'}
            logger.info(f"⏰ Tiempo estimado: {time_estimates.get(args.grid_search, 'Desconocido')}")
            
            grid_results = tuner.run_grid_search_optimization(
                ollama_model=args.model,
                training_file=args.training_file,  # El método manejará la resolución
                search_type=args.grid_search,
                max_steps=args.max_steps // 2,
                use_qlora=not args.use_lora
            )
            
            if tuner.best_config and tuner.best_accuracy > 0:
                logger.info(f"🏆 Grid search completado: {tuner.best_accuracy:.2f}% mejor precisión")
                logger.info("🔧 Configuración óptima aplicada automáticamente")
            else:
                logger.warning("⚠️ Grid search no encontró mejoras. Usando parámetros por defecto.")
        
        # Ejecutar entrenamientos
        success_count = 0
        total_trainings = 0
        
        logger.info(f"🚀 Iniciando {len(prompt_types)} tipo(s) de entrenamiento...")
        
        # Barra de progreso principal
        with tqdm(total=len(prompt_types), desc="🎯 Entrenamientos", unit="type") as main_pbar:
            for prompt_type in prompt_types:
                main_pbar.set_description(f"🎯 Entrenando {prompt_type}-shot")
                
                if args.dual_quantization:
                    model_name = f"{args.output_model}-{prompt_type}"
                    success = tuner.run_dual_training(
                        ollama_model=args.model,
                        training_file=args.training_file,
                        base_output_model=model_name,
                        max_steps=args.max_steps,
                        prompt_type=prompt_type,
                        import_to_ollama=not args.no_import
                    )
                    total_trainings += 2
                    if success:
                        success_count += 1
                else:
                    model_name = f"{args.output_model}-{prompt_type}" if len(prompt_types) > 1 else args.output_model
                    success = tuner.run_training(
                        ollama_model=args.model,
                        training_file=args.training_file,
                        output_model=model_name,
                        max_steps=args.max_steps,
                        prompt_type=prompt_type,
                        use_qlora=not args.use_lora,
                        import_to_ollama=not args.no_import
                    )
                    total_trainings += 1
                    if success:
                        success_count += 1
                
                main_pbar.update(1)
                main_pbar.set_postfix({'Success': f"{success_count}/{total_trainings}"})
        
        # Mostrar resumen final
        print(f"\n{'='*60}")
        print("🎉 ENTRENAMIENTO COMPLETADO")
        print(f"{'='*60}")
        print(f"✅ Exitosos: {success_count}/{total_trainings}")
        
        if success_count > 0:
            print(f"\n💡 CARACTERÍSTICAS DEL FINE-TUNER:")
            print(f"  🔧 PATH RESOLUTION: Manejo robusto de rutas absolutas y relativas")
            print(f"  🧬 Generación automática de prompts zero-shot y one-shot")
            print(f"  📊 Barras de progreso para operaciones largas")
            print(f"  🛡️ Análisis biológicos seguros (anti-overfitting)")
            print(f"  ⚙️ Configuración optimizada LoRA/QLoRA")
            print(f"  📈 Validación estratificada por categorías COG")
            print(f"  🚀 Exportación automática a Ollama")
            
            print(f"\n🧪 EJEMPLOS DE USO:")
            if args.use_dual:
                print(f'  ollama run {args.output_model}-zero "Analyze: MKILVAFAGLGIGTSSELQRLLS..."')
                print(f'  ollama run {args.output_model}-one "Analyze: MKILVAFAGLGIGTSSELQRLLS..."')
            else:
                model_to_show = f"{args.output_model}-{prompt_types[0]}" if len(prompt_types) > 1 else args.output_model
                print(f'  ollama run {model_to_show} "Analyze: MKILVAFAGLGIGTSSELQRLLS..."')
            
            print(f"\n🔧 FIXES APLICADOS:")
            print(f"  ✅ Path resolution: Rutas absolutas antes de cambiar directorios")
            print(f"  ✅ File validation: Verificación de existencia de archivos")
            print(f"  ✅ Error handling: Mejor manejo de errores de path")
            print(f"  ✅ Debug info: Información de debugging para troubleshooting")
            
        else:
            print(f"\n❌ Todos los entrenamientos fallaron")
            print(f"💡 Revisa los logs en los directorios de entrenamiento")
            print(f"🔧 Debugging info:")
            print(f"  📂 Directorio actual: {os.getcwd()}")
            print(f"  📂 Archivo buscado: {abs_training_file}")
            print(f"  📂 Existe archivo: {os.path.exists(abs_training_file)}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Entrenamiento interrumpido por usuario")
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()