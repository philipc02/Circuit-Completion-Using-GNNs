import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from sklearn.metrics import roc_auc_score,f1_score, accuracy_score
from sklearn.utils import shuffle
from dataloader import DataLoader, multitask_collate
import time


def train_multitask_epoch(model, optimizer, loader, device, lambda_node=1.0, lambda_edge=1.0):
    # Train one epoch with both component classification and link prediction tasks
    model.train()
    
    total_loss = 0
    total_node_loss = 0
    total_edge_loss = 0
    num_batches = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass for both tasks
        class_output, edge_scores = model(data, task='both')
        
        # Component classification loss
        node_loss = F.cross_entropy(class_output, data.y.view(-1))
        
        # Link prediction loss
        if hasattr(data, 'candidate_edges') and hasattr(data, 'edge_labels'):
            edge_loss = F.binary_cross_entropy(edge_scores, data.edge_labels.float())
        else:
            edge_loss = torch.tensor(0.0).to(device)
        
        # Combined loss
        loss = lambda_node * node_loss + lambda_edge * edge_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_node_loss += node_loss.item()
        total_edge_loss += edge_loss.item() if isinstance(edge_loss, torch.Tensor) else 0
        num_batches += 1
    
    return {
        'total_loss': total_loss / num_batches,
        'node_loss': total_node_loss / num_batches,
        'edge_loss': total_edge_loss / num_batches
    }


def eval_multitask(model, loader, device):
    # Evaluation
    model.eval()
    
    all_preds = []
    all_labels = []
    all_edge_preds = []
    all_edge_labels = []
    
    total_loss = 0
    total_node_loss = 0
    total_edge_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # Get predictions
            class_output, edge_scores = model(data, task='both')
            
            # Classification metrics
            node_loss = F.cross_entropy(class_output, data.y.view(-1), reduction='sum')
            preds = class_output.max(1)[1]
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.view(-1).cpu().numpy())
            total_node_loss += node_loss.item()
            
            # Link prediction metrics
            if hasattr(data, 'candidate_edges') and hasattr(data, 'edge_labels'):
                edge_loss = F.binary_cross_entropy(edge_scores, data.edge_labels.float(), reduction='sum')
                all_edge_preds.extend(edge_scores.cpu().numpy())
                all_edge_labels.extend(data.edge_labels.cpu().numpy())
                total_edge_loss += edge_loss.item()
            else:
                edge_loss = torch.tensor(0.0)
            
            
            num_batches += 1
    
    # Calculate metrics
    results = {
        'node_acc': accuracy_score(all_labels, all_preds),
        'node_f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        'node_f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'node_loss': total_node_loss / len(loader.dataset),
    }
    
    if len(all_edge_preds) > 0:
        all_edge_preds_binary = (np.array(all_edge_preds) > 0.5).astype(int)  # any score above 50% means edge should exist based on prediction
        results['edge_acc'] = accuracy_score(all_edge_labels, all_edge_preds_binary)
        results['edge_f1'] = f1_score(all_edge_labels, all_edge_preds_binary, average='binary')
        results['edge_auc'] = roc_auc_score(all_edge_labels, all_edge_preds)
        results['edge_loss'] = total_edge_loss / len(all_edge_labels)
    else:
        results['edge_acc'] = 0
        results['edge_f1'] = 0
        results['edge_auc'] = 0
        results['edge_loss'] = 0
    
    return results, all_preds, all_labels


def train_multitask_fegin(dataset, dataset_name, model, epochs, batch_size, lr, 
                          lr_decay_factor, lr_decay_step_size, weight_decay, 
                          device, logger=None, tracker=None, representation=None,
                          lambda_node=1.0, lambda_edge=1.0):  # loss for component classification adn link prediction currently being weighted evenly
    # Training loop for multi-task FEGIN

    print(f"Training Multi-Task FEGIN on {representation} representation")
    print(f"Loss weights: lamda_node ={lambda_node}, lambda_edge ={lambda_edge}")
    
    train_dataset = [d for d in dataset if d.set == 'train']
    test_dataset = [d for d in dataset if d.set == 'test']
    
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    train_dataset = shuffle(train_dataset, random_state=42)
    
    t_start = time.perf_counter()
    
    best_node_f1 = 0
    best_edge_f1 = 0
    best_combined_score = 0
    best_epoch = 0
    training_log = []
    best_preds, best_labels = None, None
    
    num_iterations = 10
    iteration_results = {'node_f1': [], 'edge_f1': [], 'combined': []}
    
    for iteration in range(num_iterations):
        print(f'##################ITERATION #{iteration} ****************************************')
        
        if tracker:
            tracker.start_new_iteration(iteration)
        
        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=multitask_collate)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=multitask_collate)
        
        for epoch in range(epochs):
            train_metrics = train_multitask_epoch( model, optimizer, train_loader, device, lambda_node, lambda_edge)
            
            test_metrics, epoch_preds, epoch_labels = eval_multitask(model, test_loader, device)
            
            # Combined score (weighted average of node and edge F1)
            combined_score = 0.7 * test_metrics['node_f1_weighted'] + 0.3 * test_metrics['edge_f1']
            
            log = (f"Epoch {epoch:3d}|"
                   f"Train loss: {train_metrics['total_loss']:.4f}"
                   f"(Node: {train_metrics['node_loss']:.4f}, Edge: {train_metrics['edge_loss']:.4f})|"
                   f"Test - Node f1: {test_metrics['node_f1_weighted']:.4f},"
                   f"Edge f1: {test_metrics['edge_f1']:.4f},"
                   f"Combined: {combined_score:.4f}")
            
            print(log)
            training_log.append(log)
            
            if tracker:
                tracker.log_metrics(epoch, train_metrics['node_loss'], test_metrics['node_loss'], test_metrics['node_acc'], test_metrics['node_f1_weighted'])
                tracker.log_custom_metric('edge_f1', test_metrics['edge_f1'], epoch)
                tracker.log_custom_metric('edge_auc', test_metrics['edge_auc'], epoch)
                tracker.log_custom_metric('combined_score', combined_score, epoch)
            
            # Save best model
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_node_f1 = test_metrics['node_f1_weighted']
                best_edge_f1 = test_metrics['edge_f1']
                best_epoch = epoch
                best_preds = epoch_preds
                best_labels = epoch_labels
                
                if tracker:
                    tracker.save_model(model, f"best_multitask_model_iter_{iteration}.pth")
            
            if epoch % lr_decay_step_size == 0 and epoch > 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']
        
        iteration_results['node_f1'].append(best_node_f1)
        iteration_results['edge_f1'].append(best_edge_f1)
        iteration_results['combined'].append(best_combined_score)
    
    if tracker:
        tracker.save_training_log(training_log)
        tracker.metrics['best_epoch'] = best_epoch
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    t_end = time.perf_counter()
    duration = t_end - t_start
    
    print('*********************TRAINING COMPLETE**********************')
    print(f'Duration: {duration:.2f} seconds')
    print(f'Best Epoch: {best_epoch}')
    print(f'Best Node F1: {best_node_f1:.4f}')
    print(f'Best Edge F1: {best_edge_f1:.4f}')
    print(f'Best Combined Score: {best_combined_score:.4f}')
    print(f'Across {num_iterations} iterations:')
    print(f'Node F1: {np.mean(iteration_results["node_f1"]):.4f} ± {np.std(iteration_results["node_f1"]):.4f}')
    print(f'Edge F1: {np.mean(iteration_results["edge_f1"]):.4f} ± {np.std(iteration_results["edge_f1"]):.4f}')
    print(f'Combined: {np.mean(iteration_results["combined"]):.4f} ± {np.std(iteration_results["combined"]):.4f}')
    
    node_f1_macro = f1_score(best_labels, best_preds, average="macro")
    final_acc = accuracy_score(best_labels, best_preds)
    
    return {
        'best_node_f1_weighted': best_node_f1,
        'best_edge_f1': best_edge_f1,
        'best_combined_score': best_combined_score,
        'best_acc': final_acc,
        'best_node_f1_macro': node_f1_macro,
        'predictions': np.array(best_preds),
        'labels': np.array(best_labels),
        'iteration_results': iteration_results
    }