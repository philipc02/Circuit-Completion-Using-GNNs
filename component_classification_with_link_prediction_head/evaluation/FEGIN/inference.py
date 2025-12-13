import torch
import numpy as np
import networkx as nx
import pickle
import os
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from kernel.multitask_FEGIN import MultiTaskFEGIN
from kernel.multitask_dataset import MultiTaskCircuitDataset
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve


def load_trained_model(model_path, dataset, device='cuda'):
    model = MultiTaskFEGIN(
        dataset, 
        num_layers=4,
        hidden=32,
        emb_size=128,
        use_z=False,
        use_rd=False,
        lambda_node=1.0,
        lambda_edge=1.0
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def predict_component_completion(model, original_graph, representation='component_pin_net', device='cuda'):
    print("********************COMPONENT COMPLETION INFERENCE***********************")
    
    # Step 1: Create masked graph
    # For demonstration: remove one component at time and predict
    
    component_nodes = [node for node, attr in original_graph.nodes(data=True) 
                      if attr.get('type') == 'component']
    
    if not component_nodes:
        print("No component nodes found in the graph!")
        return
    
    results = []
    
    for target_component in component_nodes[:3]:
        print(f"\n---Predicting component type at node: {target_component}---")
        
        actual_type = original_graph.nodes[target_component].get('comp_type', 'Unknown')
        print(f"Actual component type: {actual_type}")
        
        G_masked = original_graph.copy()
        removed_nodes = [target_component]
        
        # Remove pins if using pin-based representation
        if representation in ["component_pin", "component_pin_net"]:
            pin_nodes = [node for node in G_masked.neighbors(target_component) 
                        if G_masked.nodes[node].get('type') == 'pin']
            removed_nodes.extend(pin_nodes)
        
        G_masked.remove_nodes_from(removed_nodes)
        G_masked.remove_nodes_from(list(nx.isolates(G_masked)))
        
        if G_masked.number_of_nodes() < 2:
            print("  Not enough nodes after masking, skipping...")
            continue

        # Step 2: Get true connections for evaluation
        true_connections = get_component_connections(original_graph, target_component, representation)
        
        # Step 3: Convert masked graph to PyG Data
        data = convert_graph_to_pyg(G_masked, representation)
        if data is None:
            continue
        
        # Step 4: Create candidate edges from virtual node to all existing nodes
        node_mapping = {node: i for i, node in enumerate(G_masked.nodes())}
        candidate_edges = []

        VIRTUAL_NODE_IDX = -1
        
        for node in G_masked.nodes():
            idx = node_mapping[node]
            candidate_edges.append([VIRTUAL_NODE_IDX, idx])
        
        if candidate_edges:
            candidate_edges_tensor = torch.tensor(candidate_edges, dtype=torch.long).t()
            candidate_edges_list = [candidate_edges_tensor]
        else:
            candidate_edges_list = [torch.empty((2, 0), dtype=torch.long)]
        
        # Step 5: Prepare batch (single graph)
        data_batch = Batch.from_data_list([data])
        data_batch = data_batch.to(device)
        
        # Step 5: Run inference
        with torch.no_grad():
            class_output = model(data_batch, task='classification')
            predicted_class = class_output.max(1)[1].item()
            
            class_to_type = {0: 'R', 1: 'C', 2: 'V', 3: 'X'}
            predicted_type = class_to_type.get(predicted_class, 'Unknown')
            
            print(f"  Predicted component type: {predicted_type}")
            print(f"  Prediction confidence: {torch.exp(class_output[0, predicted_class]):.3f}")

            edge_scores_list = model(
                data_batch, 
                candidate_edges=candidate_edges_list, 
                task='link_prediction',
                teacher_forcing=False  # Use predicted type, not true type
            )


            predicted_connections = []
            
            if edge_scores_list and len(edge_scores_list[0]) > 0:
                edge_scores = edge_scores_list[0].cpu().numpy()

                print(f"  Edge predictions shape: {edge_scores.shape}")
                print(f"  Edge scores min/max: {edge_scores.min():.3f}/{edge_scores.max():.3f}")
                print(f"  Edge scores mean/std: {edge_scores.mean():.3f}/{edge_scores.std():.3f}")

                if len(edge_scores) != len(node_mapping):
                    print(f"  WARNING: Expected {len(node_mapping)} scores, got {len(edge_scores)}")

                all_labels = []
                node_list = list(node_mapping.keys())
                
                for node in node_list:
                    is_true_connection = node in true_connections
                    all_labels.append(1 if is_true_connection else 0)
                
                all_labels = np.array(all_labels)
                
                # Calculate auc
                try:
                    auc_score = roc_auc_score(all_labels, edge_scores)
                    print(f"  AUC-ROC: {auc_score:.3f}")
                except ValueError as e:
                    print(f"  AUC-ROC calculation failed: {e}")
                    auc_score = 0.5

                precision, recall, thresholds = precision_recall_curve(all_labels, edge_scores)
    
                # Find threshold that maximizes f1
                f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
                optimal_idx = np.argmax(f1_scores) if len(f1_scores) > 0 else 0
                optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
                
                predicted_binary = (edge_scores > optimal_threshold).astype(int)

                tp = ((predicted_binary == 1) & (all_labels == 1)).sum()
                fp = ((predicted_binary == 1) & (all_labels == 0)).sum()
                fn = ((predicted_binary == 0) & (all_labels == 1)).sum()
                
                precision_opt = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall_opt = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_opt = 2 * precision_opt * recall_opt / (precision_opt + recall_opt + 1e-10) if (precision_opt + recall_opt) > 0 else 0
                
                print(f"  Optimal threshold: {optimal_threshold:.3f}")
                print(f"  F1 at optimal threshold: {f1_opt:.3f}")
                print(f"  Precision/Recall at optimal: {precision_opt:.3f}/{recall_opt:.3f}")
                
                # Get top-k connection candidates
                k = min(10, len(edge_scores))
                top_indices = np.argsort(edge_scores)[-k:][::-1]
                
                print(f"  Top {k} connection candidates:")
                for rank, idx in enumerate(top_indices, 1):
                    node_list = list(node_mapping.keys())
                    original_node = node_list[idx]

                    node_type = G_masked.nodes[original_node].get('type', 'unknown')
                    node_comp_type = G_masked.nodes[original_node].get('comp_type', '')
                    is_true_connection = original_node in true_connections
                    truth_label = "✓" if is_true_connection else "✗"
                    print(f"    {rank}. Node {original_node} ({node_type}{'/'+node_comp_type if node_comp_type else ''}) "
                          f"score = {edge_scores[idx]:.3f} {truth_label}")
                
                # threshold = 0.5
                # Statistical threshold (mean + std)
                threshold = edge_scores.mean() + 0.5 * edge_scores.std()
                print(f"  Statistical threshold: {threshold:.3f}")
                predicted_connections = [node_list[i] for i, score in enumerate(edge_scores) if score > threshold]
                
                # True connections that are still in the masked graph
                valid_true_conn = [c for c in true_connections if c in node_mapping]
                
                tp = len(set(predicted_connections) & set(valid_true_conn))  # true positives
                fp = len(set(predicted_connections) - set(valid_true_conn))  # false positives
                fn = len(set(valid_true_conn) - set(predicted_connections))  # false negatices
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"  Link Prediction Stats:")
                print(f"    True connections: {len(valid_true_conn)}")
                print(f"    Predicted connections: {len(predicted_connections)}")
                print(f"    True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")
                print(f"    Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            
            # Check if prediction for component classification is correct
            correct_type = (predicted_type == actual_type)
            results.append({
                'target_node': target_component,
                'actual_type': actual_type,
                'predicted_type': predicted_type,
                'correct': correct_type,
                'true_connections': list(true_connections),
                'predicted_connections': predicted_connections,
                'precision': precision if 'precision' in locals() else 0,
                'recall': recall if 'recall' in locals() else 0,
                'f1': f1 if 'f1' in locals() else 0
            })
    
    # Summary
    print("*******************INFERENCE SUMMARY*******************")
    correct_count = sum(1 for r in results if r['correct'])
    accuracy = correct_count / len(results) if results else 0
    print(f"Component type accuracy: {accuracy:.1%} ({correct_count}/{len(results)})")

    if results:
        avg_precision = np.mean([r['precision'] for r in results])
        avg_recall = np.mean([r['recall'] for r in results])
        avg_f1 = np.mean([r['f1'] for r in results])
        
        print(f"Link Prediction Average:")
        print(f"  Precision: {avg_precision:.3f}")
        print(f"  Recall: {avg_recall:.3f}") 
        print(f"  F1: {avg_f1:.3f}")
    
    return results

def get_component_connections(G, comp_node, representation):
    connections = set()
    
    if representation in ["component_component", "component_net"]:
        connections = set(G.neighbors(comp_node))
    elif representation in ["component_pin", "component_pin_net"]:
        pin_nodes = [n for n in G.neighbors(comp_node) 
                    if G.nodes[n].get('type') == 'pin']
        for pin in pin_nodes:
            for neighbor in G.neighbors(pin):
                if neighbor != comp_node:
                    connections.add(neighbor)
    
    return connections

def convert_graph_to_pyg(G, representation):
    if G.number_of_nodes() == 0:
            return None
        
    # Node features
    node_features = []
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    
    node_features = get_node_features(G, representation)
    
    try:
        x = torch.tensor(np.array(node_features), dtype=torch.float)
    except Exception as e:
        # If conversion fails, dump debug info to help locate ragged vectors.
        print("ERROR: failed to convert node_features list to numpy array. Debug info:")
        for i, feat in enumerate(node_features):
            try:
                print(f"  node {i}: type={type(feat)}, shape={getattr(feat, 'shape', None)}")
            except Exception:
                print(f"  node {i}: unable to inspect feature")
        raise

    if representation == "component_component":
        # Edges
        edges = []
        for u, v, attr in G.edges(data=True):
            u_idx = node_mapping[u]
            v_idx = node_mapping[v]
            edges.append((u_idx, v_idx))
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index)
    elif representation == "component_net":            
        # Edges
        edges = []
        for u, v, attr in G.edges(data=True):
            u_idx = node_mapping[u]
            v_idx = node_mapping[v]
            edges.append((u_idx, v_idx))
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index)
    elif representation == "component_pin":
        edges = []
        edge_attrs = []
        
        for u, v, attr in G.edges(data=True):
            u_idx = node_mapping[u]
            v_idx = node_mapping[v]
            edges.append((u_idx, v_idx))
            
            edge_kind = attr.get('kind', '')
            if edge_kind == 'internal':
                edge_attrs.append([1, 0])  # Internal
            elif edge_kind == 'external':
                edge_attrs.append([0, 1])  # External
            else:
                edge_attrs.append([0, 0])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 2), dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    elif representation == "component_pin_net":
        # Edges with attributes
        edges = []
        edge_attrs = []
        
        for u, v, attr in G.edges(data=True):
            u_idx = node_mapping[u]
            v_idx = node_mapping[v]
            edges.append((u_idx, v_idx))
            
            edge_kind = attr.get('kind', '')
            if edge_kind == 'component_connection':  # internal
                edge_attrs.append([1, 0])
            elif edge_kind == 'net_connection':  # external
                edge_attrs.append([0, 1])
            else:
                edge_attrs.append([0, 0])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 2), dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data

def get_node_features(G, representation):
        if G.number_of_nodes() == 0:
                return None
        

        if representation == "component_component":
            # Node features (component type and normalized node degree)
            node_features = []
            degrees = dict(G.degree())
            
            for node, attr in G.nodes(data=True):
                node_type = attr.get('type', '')
                comp_type = attr.get('comp_type', '')
                feat = np.zeros(6, dtype=np.float32)
                
                feat[0] = 1.0  # node type: component
                feat[1] = degrees[node] / 10.0
                if comp_type in ['R', 'C', 'V', 'X']:
                    comp_idx = ['R', 'C', 'V', 'X'].index(comp_type)
                    feat[2 + comp_idx] = 1.0
                
                node_features.append(feat)
        elif representation == "component_net":            
            # Node features (node type, component type, normalized node degree)
            node_features = []
            node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
            degrees = dict(G.degree())
            
            for node, attr in G.nodes(data=True):
                node_type = attr.get('type', '')
                comp_type = attr.get('comp_type', '')
                
                feat = np.zeros(8, dtype=np.float32)
                
                if node_type == 'component':
                    feat[0] = 1.0  # node type: component
                    feat[1] = degrees[node] / 10.0
                    if comp_type in ['R', 'C', 'V', 'X']:
                        comp_idx = ['R', 'C', 'V', 'X'].index(comp_type)
                        feat[2 + comp_idx] = 1.0
                else:  # net node
                    feat[6] = 1.0  # node type: net
                    feat[7] = degrees[node] / 20.0
                                
                node_features.append(feat)
        elif representation == "component_pin":
            # Node features with edge attributes
            node_features = []
            node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
            degrees = dict(G.degree())
            
            for node, attr in G.nodes(data=True):
                node_type = attr.get('type', '')
                comp_type = attr.get('comp_type', '')
                pin_type = attr.get('pin', '')
                
                feat = np.zeros(16, dtype=np.float32) # bigger feature vector for encoding pin type
                
                if node_type == 'component':
                    feat[0] = 1.0  # node type: component
                    feat[1] = degrees[node] / 10.0
                    if comp_type in ['R', 'C', 'V', 'X']:
                        comp_idx = ['R', 'C', 'V', 'X'].index(comp_type)
                        feat[2 + comp_idx] = 1.0
                elif node_type == 'pin':
                    feat[6] = 1.0  # node type: pin
                    feat[7] = degrees[node] / 5.0
                    if pin_type in ['1', '2', 'pos', 'neg', 'p']:
                        pin_types = ['1', '2', 'pos', 'neg', 'p']
                        pin_idx = pin_types.index(pin_type)
                        feat[8 + pin_idx] = 1.0
                
                node_features.append(feat)
        elif representation == "component_pin_net":
            # Node features
            node_features = []
            node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
            degrees = dict(G.degree())
            
            for node, attr in G.nodes(data=True):
                node_type = attr.get('type', '')
                comp_type = attr.get('comp_type', '')
                pin_type = attr.get('pin', '')
                
                feat = np.zeros(16, dtype=np.float32)  # bigger feature vector for encoding pin type

                if node_type == 'component':
                    feat[0] = 1.0  # node type: component
                    feat[1] = degrees[node] / 10.0
                    if comp_type in ['R', 'C', 'V', 'X']:
                        comp_idx = ['R', 'C', 'V', 'X'].index(comp_type)
                        feat[2 + comp_idx] = 1.0
                elif node_type == 'pin':
                    feat[6] = 1.0  # node type: pin
                    feat[7] = degrees[node] / 5.0
                    if pin_type in ['1', '2', 'pos', 'neg', 'p']:
                        pin_idx = ['1', '2', 'pos', 'neg', 'p'].index(pin_type)
                        feat[8 + pin_idx] = 1.0
                elif node_type == 'net':
                    feat[13] = 1.0  # node type: net
                    feat[14] = degrees[node] / 20.0

                node_features.append(feat)
        
        return node_features

def demo_with_sample_circuit(model_path, device='cuda'):
    print("******************DEMO: COMPONENT COMPLETION ON SAMPLE CIRCUIT****************")
    
    # Load sample dataset to get model parameters
    dataset = MultiTaskCircuitDataset(
        root="data/",
        name="ltspice_demos",
        representation="component_pin_net",
        h=2,
        max_nodes_per_hop=None,
        node_label="spd",
        use_rd=False,
        neg_sampling_ratio=1.0
    )
    
    model = load_trained_model(model_path, dataset, device)
    print(f"Loaded model from: {model_path}")
    
    # Load sample circuit graph
    graph_path = "../../../graph_parsers/graphs_ltspice_demos/graphs_component_pin_net/LT1002_TA10_star_filtered.gpickle"
    
    if os.path.exists(graph_path):
        with open(graph_path, 'rb') as f:
            sample_graph = pickle.load(f)
        
        print(f"\nLoaded sample circuit with {sample_graph.number_of_nodes()} nodes, "
              f"{sample_graph.number_of_edges()} edges")
        
        # Run inference
        results = predict_component_completion(
            model=model,
            original_graph=sample_graph,
            representation="component_pin_net",
            device=device
        )
        
        return results

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Path to trained model
    model_path = Path("fegin_experiments") / "ltspice_demos_MultiTaskFEGIN__20251213001007_20251213_001007" / "models" / "best_multitask_model_iter_0.pth"
    demo_with_sample_circuit(model_path, device)
