import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from sklearn.metrics import roc_auc_score,f1_score, accuracy_score, precision_score, recall_score
from sklearn.utils import shuffle
from dataloader import DataLoader, multitask_collate, multitask_dual_collate
import time


def train_multitask_epoch(model, optimizer, loader, device, lambda_node=1.0, lambda_edge=1.0):
    # Train one epoch with both component classification and link prediction tasks
    model.train()
    
    total_loss = 0
    total_node_loss = 0
    total_edge_loss = 0
    num_batches = 0

    for batch in loader:
        if isinstance(batch, dict):
            # Classification loss
            class_data, _, _ = batch['classification']
            class_data = class_data.to(device)
            
            # Forward for classification
            class_output = model(class_data, task='classification')
            node_loss = F.cross_entropy(class_output, class_data.y.view(-1))
            
            # Pin prediction loss
            pin_data, cand_edges, edge_labels, pin_positions = batch['pin_prediction']
            edge_loss = torch.tensor(0.0, device=device)
            
            if pin_data is not None:
                pin_data = pin_data.to(device)
                if cand_edges is not None:
                    cand_edges = [ce.to(device) if ce is not None else None for ce in candidate_edges_list]
                    edge_labels = [el.to(device) for el in edge_labels]
                    pin_positions = pin_positions.to(device)
                
                # Forward for pin predictions
                edge_scores_list = model(
                    pin_data, 
                    candidate_edges=cand_edges, 
                    task='link_prediction',
                    pin_position=pin_positions
                )
                
                # Calculate edge loss
                num_edges = 0
                for edge_scores, edge_labels_batch in zip(edge_scores_list, edge_labels):
                    if len(edge_scores) > 0 and len(edge_labels_batch) > 0:
                        edge_loss += F.binary_cross_entropy(edge_scores, edge_labels_batch.float(), reduction='sum')
                        num_edges += len(edge_scores)
                
                if num_edges > 0:
                    edge_loss = edge_loss / num_edges
        elif isinstance(batch, tuple):
            data, candidate_edges_list, edge_labels_list = batch

            data = data.to(device)
            if candidate_edges_list is not None:
                # Move each tensor in the list to device
                candidate_edges_list = [ce.to(device) if ce is not None else None for ce in candidate_edges_list]
                edge_labels_list = [el.to(device) for el in edge_labels_list]
            
            optimizer.zero_grad()
            
            # Forward pass for both tasks
            class_output, edge_scores_list = model(data, candidate_edges=candidate_edges_list, task='both', teacher_forcing=True)
            
            # Component classification loss
            node_loss = F.cross_entropy(class_output, data.y.view(-1))
            
            # Link prediction loss
            if edge_labels_list is not None and edge_scores_list is not None:
                # Combine all edge losses from the batch
                edge_loss = torch.tensor(0.0, device=device)
                num_edges = 0
                
                for edge_scores, edge_labels in zip(edge_scores_list, edge_labels_list):
                    if len(edge_scores) > 0 and len(edge_labels) > 0:
                        edge_loss += F.binary_cross_entropy(edge_scores, edge_labels.float(), reduction='sum')
                        num_edges += len(edge_scores)
                
                if num_edges > 0:
                    edge_loss = edge_loss / num_edges
            else:
                edge_loss = torch.tensor(0.0).to(device)
        else:
            data = batch
            candidate_edges_list = getattr(data, 'candidate_edges', None)
            edge_labels_list = getattr(data, 'edge_labels', None)
        
            data = data.to(device)
            if candidate_edges_list is not None:
                # Move each tensor in the list to device
                candidate_edges_list = [ce.to(device) if ce is not None else None for ce in candidate_edges_list]
                edge_labels_list = [el.to(device) for el in edge_labels_list]
            
            optimizer.zero_grad()
            
            # Forward pass for both tasks
            class_output, edge_scores_list = model(data, candidate_edges=candidate_edges_list, task='both', teacher_forcing=True)
            
            # Component classification loss
            node_loss = F.cross_entropy(class_output, data.y.view(-1))
            
            # Link prediction loss
            if edge_labels_list is not None and edge_scores_list is not None:
                # Combine all edge losses from the batch
                edge_loss = torch.tensor(0.0, device=device)
                num_edges = 0
                
                for edge_scores, edge_labels in zip(edge_scores_list, edge_labels_list):
                    if len(edge_scores) > 0 and len(edge_labels) > 0:
                        edge_loss += F.binary_cross_entropy(edge_scores, edge_labels.float(), reduction='sum')
                        num_edges += len(edge_scores)
                
                if num_edges > 0:
                    edge_loss = edge_loss / num_edges
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
        for batch in loader:
            if isinstance(batch, dict):
                # Classification evaluation
                class_data, _, _ = batch['classification']
                class_data = class_data.to(device)
                
                class_output = model(class_data, task='classification')
                class_preds = class_output.max(1)[1]
                
                all_preds.extend(class_preds.cpu().numpy())
                all_labels.extend(class_data.y.view(-1).cpu().numpy())
                
                # Pin prediction evaluation
                pin_data, cand_edges, edge_labels, pin_positions = batch['pin_prediction']
                if pin_data is not None:
                    pin_data = pin_data.to(device)
                    if cand_edges is not None:
                        cand_edges = [ce.to(device) if ce is not None else None for ce in candidate_edges_list]
                        edge_labels = [el.to(device) for el in edge_labels]
                        pin_positions = pin_positions.to(device)
                    
                    edge_scores_list = model(
                        pin_data,
                        candidate_edges=cand_edges,
                        task='link_prediction',
                        pin_position=pin_positions
                    )
                    
                    for edge_scores, edge_labels_batch in zip(edge_scores_list, edge_labels):
                        if len(edge_scores) > 0 and len(edge_labels_batch) > 0:
                            all_edge_preds.extend(edge_scores.cpu().numpy())
                            all_edge_labels.extend(edge_labels_batch.cpu().numpy())
            elif isinstance(batch, tuple):
                data, candidate_edges_list, edge_labels_list = batch

                data = data.to(device)
                if candidate_edges_list is not None:
                    # Move each tensor in the list to device
                    candidate_edges_list = [ce.to(device) if ce is not None else None for ce in candidate_edges_list]
                    edge_labels_list = [el.to(device) for el in edge_labels_list]
                
                # Get predictions
                class_output, edge_scores_list = model(data, candidate_edges=candidate_edges_list, task='both', teacher_forcing=False)  # No teacher forcing during evaluation
            else:
                data = batch
                candidate_edges_list = getattr(data, 'candidate_edges', None)
                edge_labels_list = getattr(data, 'edge_labels', None)
            
                data = data.to(device)
                if candidate_edges_list is not None:
                    # Move each tensor in the list to device
                    candidate_edges_list = [ce.to(device) if ce is not None else None for ce in candidate_edges_list]
                    edge_labels_list = [el.to(device) for el in edge_labels_list]
                
                # Get predictions
                class_output, edge_scores_list = model(data, candidate_edges=candidate_edges_list, task='both', teacher_forcing=False)  # No teacher forcing during evaluation
            
            # Classification metrics
            node_loss = F.cross_entropy(class_output, data.y.view(-1), reduction='sum')
            preds = class_output.max(1)[1]
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.view(-1).cpu().numpy())
            total_node_loss += node_loss.item()
            
            # Link prediction metrics
            batch_edge_loss = torch.tensor(0.0, device=device)
            if edge_labels_list is not None and edge_scores_list is not None:
                # Combine all edge losses from the batch
                num_edges = 0
                
                for edge_scores, edge_labels in zip(edge_scores_list, edge_labels_list):
                    if len(edge_scores) > 0 and len(edge_labels) > 0:
                        batch_edge_loss += F.binary_cross_entropy(edge_scores, edge_labels.float(), reduction='sum')
                        all_edge_preds.extend(edge_scores.cpu().numpy())
                        all_edge_labels.extend(edge_labels.cpu().numpy())
                        num_edges += len(edge_scores)
                
                if num_edges > 0:
                    batch_edge_loss = batch_edge_loss / num_edges
                    total_edge_loss += batch_edge_loss.item()
            
            
            num_batches += 1
    
    # Calculate metrics
    results = {
        'node_acc': accuracy_score(all_labels, all_preds),
        'node_f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        'node_f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'node_loss': total_node_loss / len(all_labels),
    }
    
    if len(all_edge_preds) > 0:
        # Calculate multiple metrics for edge prediction
        all_edge_preds_array = np.array(all_edge_preds)
        all_edge_labels_array = np.array(all_edge_labels)
        
        # AUC-ROC, threshold independent
        try:
            edge_auc = roc_auc_score(all_edge_labels_array, all_edge_preds_array)
        except ValueError:
            # When only one class is present
            edge_auc = 0.5

        all_edge_preds_binary = (np.array(all_edge_preds) > 0.5).astype(int)  # any score above 50% means edge should exist based on prediction
        results['edge_acc'] = accuracy_score(all_edge_labels, all_edge_preds_binary)
        results['edge_precision'] = precision_score(all_edge_labels_array, all_edge_preds_binary, zero_division=0)
        results['edge_recall'] = recall_score(all_edge_labels_array, all_edge_preds_binary, zero_division=0)
        results['edge_f1'] = f1_score(all_edge_labels, all_edge_preds_binary, average='binary')
        results['edge_auc'] = edge_auc  # primary metric!
        results['edge_loss'] = total_edge_loss / len(all_edge_labels)
    else:
        results['edge_acc'] = 0
        results['edge_precision'] = 0
        results['edge_recall'] = 0
        results['edge_f1'] = 0
        results['edge_auc'] = 0.5  # random chance
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
    best_edge_auc = 0
    best_combined_score = 0
    best_epoch = 0
    training_log = []
    best_preds, best_labels = None, None
    
    num_iterations = 10
    iteration_results = {'node_f1': [], 'edge_auc': [], 'combined': []}
    
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
            
            # Combined score (weighted average of node and edge auc)
            combined_score = 0.7 * test_metrics['node_f1_weighted'] + 0.3 * test_metrics['edge_auc']
            
            log = (f"Epoch {epoch:3d}|"
                   f"Train loss: {train_metrics['total_loss']:.4f}"
                   f"(Node: {train_metrics['node_loss']:.4f}, Edge: {train_metrics['edge_loss']:.4f})|"
                   f"Test - Node f1: {test_metrics['node_f1_weighted']:.4f},"
                   f"Edge AUC: {test_metrics['edge_auc']:.4f},"
                   f"Combined: {combined_score:.4f}")
            
            print(log)
            training_log.append(log)
            
            if tracker:
                tracker.log_metrics(epoch, train_metrics['node_loss'], test_metrics['node_loss'], test_metrics['node_acc'], test_metrics['node_f1_weighted'])
                tracker.log_custom_metric('edge_f1', test_metrics['edge_f1'], epoch)
                tracker.log_custom_metric('edge_auc', test_metrics['edge_auc'], epoch)
                tracker.log_custom_metric('edge_precision', test_metrics['edge_precision'], epoch)
                tracker.log_custom_metric('edge_recall', test_metrics['edge_recall'], epoch)
                tracker.log_custom_metric('combined_score', combined_score, epoch)
            
            # Save best model
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_node_f1 = test_metrics['node_f1_weighted']
                best_edge_auc = test_metrics['edge_auc']
                best_epoch = epoch
                best_preds = epoch_preds
                best_labels = epoch_labels
                
                if tracker:
                    tracker.save_model(model, f"best_multitask_model_iter_{iteration}.pth")
            
            if epoch % lr_decay_step_size == 0 and epoch > 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']
        
        iteration_results['node_f1'].append(best_node_f1)
        iteration_results['edge_auc'].append(best_edge_auc)
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
    print(f'Best Edge AUC: {best_edge_auc:.4f}')
    print(f'Best Combined Score: {best_combined_score:.4f}')
    print(f'Across {num_iterations} iterations:')
    print(f'Node F1: {np.mean(iteration_results["node_f1"]):.4f} ± {np.std(iteration_results["node_f1"]):.4f}')
    print(f'Edge AUC: {np.mean(iteration_results["edge_auc"]):.4f} ± {np.std(iteration_results["edge_auc"]):.4f}')
    print(f'Combined: {np.mean(iteration_results["combined"]):.4f} ± {np.std(iteration_results["combined"]):.4f}')
    
    node_f1_macro = f1_score(best_labels, best_preds, average="macro")
    final_acc = accuracy_score(best_labels, best_preds)
    
    return {
        'best_node_f1_weighted': best_node_f1,
        'best_edge_auc': best_edge_auc,
        'best_combined_score': best_combined_score,
        'best_acc': final_acc,
        'best_node_f1_macro': node_f1_macro,
        'predictions': np.array(best_preds),
        'labels': np.array(best_labels),
        'iteration_results': iteration_results
    }