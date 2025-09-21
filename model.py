# Multi-Antibiotic Resistance Prediction - Sequential Version for Overnight Training
import pandas as pd
import numpy as np
#from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import xgboost as xgb
import re
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os
warnings.filterwarnings('ignore')

class MultiAntibioticPredictor:
    def __init__(self, k=6, max_features=5000, min_df=3, use_gpu=True):
        self.k = k
        self.max_features = max_features
        self.min_df = min_df
        self.use_gpu = use_gpu
        self.gpu_available = self.check_gpu_availability() if use_gpu else False
        self.vectorizer = None
        self.models = {}
        self.antibiotics = []
        self.performance_metrics = {}

    def check_gpu_availability(self):
        """Check if GPU is available for XGBoost"""
        try:
            dtrain = xgb.DMatrix([[1, 2], [3, 4]], label=[1, 0])
            params = {'tree_method': 'gpu_hist', 'gpu_id': 0, 'verbosity': 0}
            xgb.train(params, dtrain, num_boost_round=1, verbose_eval=False)
            print("GPU is available for XGBoost training!")
            return True
        except Exception as e:
            print(f"GPU not available for XGBoost: {e}")
            print("Falling back to CPU training with optimization...")
            return False

    def extract_kmers(self, dna_sequence):
        """Extract k-mers from DNA sequence"""
        sequence = re.sub(r'\s+', '', str(dna_sequence)).upper()
        kmers = []
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i+self.k]
            if all(base in 'ATGC' for base in kmer):
                kmers.append(kmer)
        return ' '.join(kmers)

    def prepare_data(self, sample_size=2000):
        """Load and prepare data"""
        print("Loading binary labels...")
        labels_df = pd.read_csv('binary_labels.csv').set_index('genome_name')

        print("Loading genome dataset...")
        ds = load_dataset("macwiatrak/bacbench-antibiotic-resistance-dna", split="train", streaming=True)

        genomes = []
        genome_names = []

        print(f"Sampling {sample_size} genomes...")
        start_time = time.time()
        for i, item in enumerate(tqdm(ds, desc="Loading genomes")):
            if i >= sample_size:
                break
            genome_names.append(item['genome_name'])
            genomes.append(item['dna_sequence'])

            # Progress update every 500 genomes
            if (i + 1) % 500 == 0:
                elapsed = time.time() - start_time
                print(f"Loaded {i+1} genomes in {elapsed/60:.1f} minutes")

        genome_df = pd.DataFrame({
            'genome_name': genome_names,
            'dna_sequence': genomes
        }).set_index('genome_name')

        data = genome_df.join(labels_df, how='inner')
        print(f"Final dataset shape: {data.shape}")

        return data

    def analyze_antibiotic_coverage(self, data):
        """Analyze which antibiotics have sufficient data"""
        print("\n=== Antibiotic Data Coverage ===")
        antibiotic_stats = []

        antibiotic_cols = [col for col in data.columns if col != 'dna_sequence']

        for antibiotic in antibiotic_cols:
            valid_mask = ~data[antibiotic].isna()
            n_valid = valid_mask.sum()

            if n_valid > 0:
                n_resistant = (data[antibiotic] == 1).sum()
                n_susceptible = (data[antibiotic] == 0).sum()
                resistance_rate = n_resistant / n_valid if n_valid > 0 else 0

                antibiotic_stats.append({
                    'antibiotic': antibiotic,
                    'total_samples': n_valid,
                    'resistant': n_resistant,
                    'susceptible': n_susceptible,
                    'resistance_rate': resistance_rate,
                    'trainable': n_valid >= 20 and min(n_resistant, n_susceptible) >= 5
                })

        stats_df = pd.DataFrame(antibiotic_stats).sort_values('total_samples', ascending=False)

        print(f"Total antibiotics: {len(stats_df)}")
        print(f"Trainable antibiotics: {stats_df['trainable'].sum()}")
        print("\nTop 15 antibiotics by sample size:")
        print(stats_df.head(15)[['antibiotic', 'total_samples', 'resistant', 'susceptible', 'resistance_rate', 'trainable']].round(3))

        return stats_df

    def train_all_models(self, data, min_samples=20):
        """Train models for all viable antibiotics with checkpoint saving"""

        # Check trainable antibiotics
        antibiotic_cols = [col for col in data.columns if col != 'dna_sequence']
        trainable_antibiotics = []

        print(f"Checking {len(antibiotic_cols)} antibiotics for training viability...")

        for antibiotic in antibiotic_cols:
            valid_mask = ~data[antibiotic].isna()
            n_valid = int(valid_mask.sum())
            if n_valid >= min_samples:
                y_temp = data[antibiotic][valid_mask].values.astype(int)
                if len(np.unique(y_temp)) >= 2 and min(np.bincount(y_temp)) >= 5:
                    trainable_antibiotics.append(antibiotic)
                    print(f"  {antibiotic}: {n_valid} samples - TRAINABLE")

        if not trainable_antibiotics:
            print("\nNO TRAINABLE ANTIBIOTICS FOUND!")
            return

        # Sequential k-mer extraction with progress updates
        print(f"\nExtracting {self.k}-mers from {len(data)} genomes (sequential processing)...")
        print("This will take 2-3 hours. Progress will be saved periodically.")

        start_time = time.time()
        kmer_docs = []

        for i, seq in enumerate(tqdm(data['dna_sequence'], desc="Extracting k-mers")):
            kmer_doc = self.extract_kmers(seq)
            kmer_docs.append(kmer_doc)

            # Progress update and checkpoint every 200 sequences
            if (i + 1) % 200 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(data) - i - 1) / rate
                print(f"Processed {i+1}/{len(data)} genomes. "
                      f"Rate: {rate:.2f} genomes/sec. "
                      f"Est. remaining: {remaining/60:.1f} minutes")

                # Save checkpoint
                checkpoint_data = {
                    'kmer_docs': kmer_docs,
                    'progress': i + 1,
                    'total': len(data)
                }
                with open('kmer_checkpoint.pkl', 'wb') as f:
                    pickle.dump(checkpoint_data, f)

        total_time = time.time() - start_time
        print(f"K-mer extraction completed in {total_time/60:.1f} minutes")

        # TF-IDF vectorization
        print("Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df
        )
        X = self.vectorizer.fit_transform(kmer_docs)
        print(f"Feature matrix shape: {X.shape}")

        # Save vectorizer checkpoint
        with open('vectorizer_checkpoint.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)

        self.antibiotics = trainable_antibiotics
        print(f"\nTraining models for {len(trainable_antibiotics)} antibiotics...")
        print(f"Using {'GPU' if self.gpu_available else 'CPU'} acceleration")

        # Train models
        successful_models = 0
        for antibiotic in tqdm(trainable_antibiotics, desc="Training models"):
            try:
                valid_indices = ~data[antibiotic].isna()
                X_valid = X[valid_indices.values]
                y_valid = data[antibiotic][valid_indices].values.astype(int)

                unique_classes, class_counts = np.unique(y_valid, return_counts=True)
                if len(unique_classes) < 2 or min(class_counts) < 5:
                    continue

                X_train, X_test, y_train, y_test = train_test_split(
                    X_valid, y_valid, test_size=0.2, random_state=42,
                    stratify=y_valid
                )

                # Configure model with GPU or CPU
                if self.gpu_available:
                    model = xgb.XGBClassifier(
                        n_estimators=100, max_depth=6, learning_rate=0.1,
                        random_state=42, eval_metric='logloss', verbosity=0,
                        tree_method='gpu_hist', gpu_id=0, predictor='gpu_predictor'
                    )
                else:
                    model = xgb.XGBClassifier(
                        n_estimators=100, max_depth=6, learning_rate=0.1,
                        random_state=42, eval_metric='logloss', verbosity=0,
                        tree_method='hist', n_jobs=-1
                    )

                model.fit(X_train, y_train)

                y_pred_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred_proba)
                auprc = average_precision_score(y_test, y_pred_proba)

                self.models[antibiotic] = model
                self.performance_metrics[antibiotic] = {
                    'auc': auc, 'auprc': auprc, 'n_samples': len(y_valid),
                    'n_train': len(y_train), 'n_test': len(y_test),
                    'resistance_rate': y_valid.mean(), 'gpu_trained': self.gpu_available
                }
                successful_models += 1

                # Save progress after each model
                if successful_models % 5 == 0:
                    self.save_models(f'models_checkpoint_{successful_models}.pkl')

            except Exception as e:
                print(f"Failed to train model for {antibiotic}: {str(e)}")

        print(f"\nSuccessfully trained {successful_models} models using {'GPU' if self.gpu_available else 'CPU'}")

    def evaluate_all_models(self):
        """Display performance metrics for all models"""
        if not self.performance_metrics:
            print("No models trained yet!")
            return None

        metrics_df = pd.DataFrame(self.performance_metrics).T
        metrics_df = metrics_df.sort_values('auc', ascending=False)

        print("\n=== Model Performance Summary ===")
        print(f"Training method: {'GPU-accelerated' if self.gpu_available else 'CPU-optimized'}")
        print(f"Average AUC: {metrics_df['auc'].mean():.3f} ± {metrics_df['auc'].std():.3f}")
        print(f"Average AUPRC: {metrics_df['auprc'].mean():.3f} ± {metrics_df['auprc'].std():.3f}")
        print(f"Models with AUC > 0.7: {(metrics_df['auc'] > 0.7).sum()}/{len(metrics_df)}")
        print(f"Models with AUC > 0.8: {(metrics_df['auc'] > 0.8).sum()}/{len(metrics_df)}")

        print("\nTop 10 performing models:")
        display_cols = ['auc', 'auprc', 'n_samples', 'resistance_rate']
        print(metrics_df[display_cols].head(10).round(3))

        return metrics_df

    def predict_all_antibiotics(self, genome_sequence):
        """Predict resistance for all antibiotics"""
        if not self.models:
            print("No models trained!")
            return None

        kmer_seq = self.extract_kmers(genome_sequence)
        X_new = self.vectorizer.transform([kmer_seq])

        predictions = {}
        for antibiotic, model in self.models.items():
            prob = model.predict_proba(X_new)[0, 1]
            prediction = "Resistant" if prob > 0.5 else "Susceptible"
            predictions[antibiotic] = {
                'prediction': prediction,
                'resistance_probability': prob,
                'confidence': max(prob, 1-prob)
            }

        sorted_predictions = dict(sorted(
            predictions.items(),
            key=lambda x: x[1]['resistance_probability'],
            reverse=True
        ))

        return sorted_predictions

    def save_models(self, filename='multi_antibiotic_models.pkl'):
        """Save all models"""
        model_data = {
            'vectorizer': self.vectorizer,
            'models': self.models,
            'antibiotics': self.antibiotics,
            'performance_metrics': self.performance_metrics,
            'k': self.k,
            'max_features': self.max_features,
            'min_df': self.min_df,
            'use_gpu': self.use_gpu,
            'gpu_available': self.gpu_available
        }

        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Models saved to {filename}")

    @classmethod
    def load_models(cls, filename='multi_antibiotic_models.pkl'):
        """Load saved models"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)

        predictor = cls(
            k=model_data['k'],
            max_features=model_data['max_features'],
            min_df=model_data['min_df'],
            use_gpu=model_data.get('use_gpu', True)
        )

        predictor.vectorizer = model_data['vectorizer']
        predictor.models = model_data['models']
        predictor.antibiotics = model_data['antibiotics']
        predictor.performance_metrics = model_data['performance_metrics']
        predictor.gpu_available = model_data.get('gpu_available', False)

        return predictor

# Main execution function
def main():
    print("=== Multi-Antibiotic Resistance Prediction (Overnight Training) ===")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    overall_start = time.time()

    predictor = MultiAntibioticPredictor(k=6, max_features=5000, use_gpu=True)
    data = predictor.prepare_data(sample_size=2000)

    stats_df = predictor.analyze_antibiotic_coverage(data)

    if len(stats_df) > 0:
        predictor.train_all_models(data)
        metrics_df = predictor.evaluate_all_models()
        predictor.save_models()

        total_time = time.time() - overall_start
        print(f"\nTotal training time: {total_time/3600:.1f} hours")
        print(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("No data available for training.")
        metrics_df = None

    return predictor, metrics_df

if __name__ == "__main__":
    predictor, metrics = main()