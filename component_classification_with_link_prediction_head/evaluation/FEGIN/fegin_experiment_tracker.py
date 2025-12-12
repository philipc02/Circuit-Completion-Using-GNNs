import datetime
from pathlib import Path
import json
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os

class FEGINExperimentTracker:
    def __init__(self, experiment_name, dataset_name, model_name):
        self.experiment_name = experiment_name
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_dir = Path("fegin_experiments") / f"{dataset_name}_{model_name}_{experiment_name}_{self.timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': [],
            'test_acc': None, 'test_f1': None, 'test_f1_macro': None,
            'best_epoch': None, 'config': {}, 'current_iteration': 0, 'iteration_metrics': {}, 'edge_f1' : [], 'edge_auc' : [], 'combined_score' : []
        }
        
        (self.experiment_dir / "models").mkdir(exist_ok=True)
        (self.experiment_dir / "predictions").mkdir(exist_ok=True)

    def log_config(self, config_dict):
        self.metrics['config'] = config_dict
        with open(self.experiment_dir / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        with open(self.experiment_dir / "config.txt", 'w') as f:
            f.write("FEGIN Experiment Configuration\n")
            for key, value in config_dict.items():
                f.write(f"{key}: {value}\n")

    def start_new_iteration(self, iteration_num):
        # start tracking new iteration
        self.metrics['current_iteration'] = iteration_num
        self.metrics['iteration_metrics'][iteration_num] = {
            'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []
        }

    def log_metrics(self, epoch, train_loss, val_loss, val_acc, val_f1):
        current_iter = self.metrics['current_iteration']
        # store in iteration specific metrics
        self.metrics['iteration_metrics'][current_iter]['train_loss'].append(float(train_loss))
        self.metrics['iteration_metrics'][current_iter]['val_loss'].append(float(val_loss))
        self.metrics['iteration_metrics'][current_iter]['val_acc'].append(float(val_acc))
        self.metrics['iteration_metrics'][current_iter]['val_f1'].append(float(val_f1))
        
        self.metrics['train_loss'].append(float(train_loss))
        self.metrics['val_loss'].append(float(val_loss))
        self.metrics['val_acc'].append(float(val_acc))
        self.metrics['val_f1'].append(float(val_f1))
        
        self.save_metrics_to_csv()

    def log_custom_metric(self, metric, value, epoch):
        current_iter = self.metrics['current_iteration']
        # store in iteration specific metrics
        self.metrics['iteration_metrics'][current_iter][metric].append(float(value))
        self.save_metrics_to_csv()

    def log_best_scores(self, best_edge_f1, best_combined_score):
        # best scores from entire training run
        self.metrics['best_edge_f1'] = float(best_edge_f1)
        self.metrics['best_combined_score'] = float(best_combined_score)
        
        # Save to separate file
        with open(self.experiment_dir / "best_scores.json", 'w') as f:
            json.dump({
                'best_edge_f1': float(best_edge_f1),
                'best_combined_score': float(best_combined_score)
            }, f, indent=2)
        
        summary_file = self.experiment_dir / "results_summary.txt"
        if summary_file.exists():
            with open(summary_file, 'a') as f:
                f.write(f"Best Edge F1: {best_edge_f1:.4f}\n")
                f.write(f"Best Combined Score: {best_combined_score:.4f}\n")

    def save_metrics_to_csv(self):
        # global metrics
        if len(self.metrics['train_loss']) > 0:
            global_metrics_df = pd.DataFrame({
                'global_epoch': list(range(len(self.metrics['train_loss']))),
                'train_loss': self.metrics['train_loss'],
                'val_loss': self.metrics['val_loss'], 
                'val_acc': self.metrics['val_acc'],
                'val_f1': self.metrics['val_f1']
            })
            global_metrics_df.to_csv(self.experiment_dir / "metrics_global.csv", index=False)
        
        # per-iteration metrics
        for iter_num, metrics in self.metrics['iteration_metrics'].items():
            if len(metrics['train_loss']) > 0:
                iter_data ={
                    'epoch': list(range(len(metrics['train_loss']))),
                    'train_loss': metrics['train_loss'],
                    'val_loss': metrics['val_loss'],
                    'val_acc': metrics['val_acc'],
                    'val_f1': metrics['val_f1']
                }
                # Add any custom metrics if they exist
                for metric_name in ['edge_f1', 'edge_auc', 'combined_score']:
                    if metric_name in metrics and len(metrics[metric_name]) > 0:
                        iter_data[metric_name] = metrics[metric_name]
                iter_df = pd.DataFrame(iter_data)
                iter_df.to_csv(self.experiment_dir / f"metrics_iteration_{iter_num}.csv", index=False)

    def log_test_results(self, test_acc, test_f1_weighted, test_f1_macro, all_preds, all_labels, class_names):
        self.metrics['test_acc'] = float(test_acc)
        self.metrics['test_f1'] = float(test_f1_weighted)
        self.metrics['test_f1_macro'] = float(test_f1_macro)
        
        test_results = {
            'test_accuracy': float(test_acc),
            'test_f1_weighted': float(test_f1_weighted),
            'test_f1_macro': float(test_f1_macro),
            'predictions': all_preds.tolist() if hasattr(all_preds, 'tolist') else all_preds,
            'labels': all_labels.tolist() if hasattr(all_labels, 'tolist') else all_labels
        }
        
        with open(self.experiment_dir / "test_results.json", 'w') as f:
            json.dump(test_results, f, indent=2)
        
        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
        with open(self.experiment_dir / "classification_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        cm = confusion_matrix(all_labels, all_preds)
        np.save(self.experiment_dir / "confusion_matrix.npy", cm)
        
        with open(self.experiment_dir / "results_summary.txt", 'w') as f:
            f.write("FEGIN Experiment Results\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Test Accuracy: {test_acc:.4f}\n")
            f.write(f"Test F1 (Weighted): {test_f1_weighted:.4f}\n")
            f.write(f"Test F1 (Macro): {test_f1_macro:.4f}\n")
            f.write(f"Timestamp: {self.timestamp}\n")

    def save_model(self, model, name="best_model.pth"):
        model_path = self.experiment_dir / "models" / name
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.metrics['config'],
            'metrics': self.metrics
        }, model_path)

    def save_training_log(self, log_entries):
        with open(self.experiment_dir / "training_log.txt", 'w') as f:
            f.write("\n".join(log_entries))