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
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_trained_model(model_path, dataset, device='cuda'):
    model = MultiTaskFEGIN(
        dataset, 
        num_layers=4,
        hidden=32,
        emb_size=128,
        use_z=False,
        use_rd=False,
        lambda_node=1.0,
        lambda_edge=1.0,
        max_pins=2
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def predict_component_completion(model, original_graph, representation='component_pin_net', device='cuda'):
    print("********************COMPONENT COMPLETION INFERENCE***********************")
    
    # TASK 1: COMPONENT CLASSIFICATION
    
    component_nodes = [node for node, attr in original_graph.nodes(data=True) 
                      if attr.get('type') == 'component']
    
    if not component_nodes:
        print("No component nodes found in the graph!")
        return
    
    results = []
    
    for target_component in component_nodes[:10]:
        print(f"\n---Predicting component type at node: {target_component}---")
        
        actual_type = original_graph.nodes[target_component].get('comp_type', 'Unknown')
        print(f"Actual component type: {actual_type}")
        
        G_masked = original_graph.copy()
        removed_nodes = [target_component]
        
        # Remove pins if using pin-based representation
        if representation in ["component_pin", "component_pin_net"]:
            pin_nodes = [node for node in G_masked.neighbors(target_component) 
                        if G_masked.nodes[node].get('type') == 'pin']
            pin_nodes.sort(key=lambda n: original_graph.nodes[n].get('pin', ''))
            removed_nodes.extend(pin_nodes)
        
        G_masked.remove_nodes_from(removed_nodes)
        G_masked.remove_nodes_from(list(nx.isolates(G_masked)))
        
        if G_masked.number_of_nodes() < 2:
            print("  Not enough nodes after masking, skipping...")
            continue

        
        data = convert_graph_to_pyg(G_masked, representation)
        if data is None:
            continue
        
        data_batch = Batch.from_data_list([data])
        data_batch = data_batch.to(device)
        
        # Run inference
        with torch.no_grad():
            class_output = model(data_batch, task='classification')
            predicted_class = class_output.max(1)[1].item()
            
            class_to_type = {0: 'R', 1: 'C', 2: 'V', 3: 'X'}
            predicted_type = class_to_type.get(predicted_class, 'Unknown')
            
            print(f"  Predicted component type: {predicted_type}")
            print(f"  Prediction confidence: {torch.exp(class_output[0, predicted_class]):.3f}")

            correct_type = (predicted_type == actual_type)

            # TASK 2: PIN CONNECTION PREDICTION for each pin
            pin_nodes = [node for node in original_graph.neighbors(target_component) 
                        if original_graph.nodes[node].get('type') == 'pin']
            pin_nodes.sort(key=lambda n: original_graph.nodes[n].get('pin', ''))

            all_pin_results = []
            
            for pin_idx, pin_node in enumerate(pin_nodes):
                if pin_idx >= 2:  # handle up to 2 pins (max_pins, can be changed/increased)
                    break
                    
                print(f"\n  Predicting connections for pin {pin_idx} ({pin_node})")
                print(f"    Pin features: {original_graph.nodes[pin_node]}")
                
                if representation == 'component_pin_net':
                    # Get the net this pin should connect to (ground truth)
                    connected_nets = [n for n in original_graph.neighbors(pin_node) 
                                    if original_graph.nodes[n].get('type') == 'net']
                    
                    # TODO: (future) model that predicts whether pin nodes has connection or not (not only which node it connects to when it definetly has a connection)
                    if not connected_nets:
                        print(f"    Pin {pin_idx} has no connected net, skipping...")
                        continue
                        
                    target_net = connected_nets[0]
                    print(f"    True connection: pin {pin_idx} -> net {target_net}")
                    
                    # Create graph with only this pin's edges masked
                    G_pin_masked = original_graph.copy()
                    edges_to_remove = []
                    for u, v in G_pin_masked.edges():
                        if u == pin_node or v == pin_node:
                            edges_to_remove.append((u, v))
                    G_pin_masked.remove_edges_from(edges_to_remove)
                    
                    # Generate candidate edges from virtual node to all net nodes
                    node_mapping = {node: i for i, node in enumerate(G_pin_masked.nodes())}
                    candidate_edges = []

                    VIRTUAL_NODE_IDX = -1
                    
                    # Only consider net nodes as candidates
                    # TODO: is this making it very easy for the model?
                    net_nodes = [node for node in G_pin_masked.nodes() 
                                if G_pin_masked.nodes[node].get('type') == 'net']
                    
                    for net_node in net_nodes:
                        idx = node_mapping[net_node]
                        candidate_edges.append([VIRTUAL_NODE_IDX, idx])

                elif representation == 'component_pin':
                    # Predict pin-to-pin connections
                    connected_pins = []
                    for neighbor in original_graph.neighbors(pin_node):
                        if (original_graph.nodes[neighbor].get('type') == 'pin' and 
                            neighbor not in pin_nodes):  # Pin from other component
                            connected_pins.append(neighbor)
                    
                    if not connected_pins:
                        print(f"    Pin {pin_idx} has no connected pins, skipping...")
                        continue
                    
                    target_node = connected_pins[0]
                    print(f"    True connection: pin {pin_idx} -> pin {target_node}")
                    
                    # Create graph with only this pin masked
                    G_pin_masked = original_graph.copy()
                    edges_to_remove = []
                    for u, v in G_pin_masked.edges():
                        if u == pin_node or v == pin_node:
                            edges_to_remove.append((u, v))
                    G_pin_masked.remove_edges_from(edges_to_remove)
                    
                    # Get all pins (except own component's pins) as candidates
                    node_mapping = {node: i for i, node in enumerate(G_pin_masked.nodes())}
                    candidate_edges = []
                    
                    VIRTUAL_NODE_IDX = -1

                    other_pins = [node for node in G_pin_masked.nodes() 
                                 if G_pin_masked.nodes[node].get('type') == 'pin' 
                                 and node not in pin_nodes]
                    
                    for pin in other_pins:
                        idx = node_mapping[pin]
                        candidate_edges.append([VIRTUAL_NODE_IDX, idx])

                else:
                    print(f"    Representation {representation} not supported for link prediction")
                    continue
                
                if not candidate_edges:
                    print(f"    No net nodes found as candidates, skipping...")
                    continue
                
                candidate_edges_tensor = torch.tensor(candidate_edges, dtype=torch.long).t()
                
                # Convert masked graph to PyG Data
                pin_data = convert_graph_to_pyg(G_pin_masked, representation)
                if pin_data is None:
                    continue
                
                # Pin position attribute
                pin_data.pin_position = torch.tensor([pin_idx], dtype=torch.long)
                
                pin_batch = Batch.from_data_list([pin_data]).to(device)
                
                # Use predicted component type (teacher_forcing=False)
                with torch.no_grad():
                    edge_scores = model(
                        class_data=data_batch,
                        pin_data_list=[[pin_batch]],
                        candidate_edges_list=[[candidate_edges_tensor]],
                        task='both',
                        teacher_forcing=False,
                        pin_position=[[pin_idx]]
                    )[1]  # Returns value (class_output, all_edge_scores)
                    
                    # Extract scores
                    if edge_scores and edge_scores[0] and len(edge_scores[0]) > 0:
                        pin_edge_scores = edge_scores[0][0].cpu().numpy()
                    else:
                        print(f"    No edge scores returned for pin {pin_idx}")
                        continue

                all_labels = []
                if representation == 'component_pin_net':
                    for net_node in net_nodes:
                        is_true_connection = (net_node == target_net)
                        all_labels.append(1 if is_true_connection else 0)
                else:  # component_pin
                    for other_pin in other_pins:
                        is_true_connection = (other_pin == target_node)
                        all_labels.append(1 if is_true_connection else 0)
                
                all_labels = np.array(all_labels)
                
                # Calculate auc
                try:
                    auc_score = roc_auc_score(all_labels, pin_edge_scores)
                    print(f"  AUC-ROC: {auc_score:.3f}")
                except ValueError as e:
                    print(f"  AUC-ROC calculation failed: {e}")
                    auc_score = 0.5

                precision, recall, thresholds = precision_recall_curve(all_labels, pin_edge_scores)
    
                if len(thresholds) > 0:
                    # Find threshold that maximizes f1
                    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
                    optimal_idx = np.argmax(f1_scores)
                    optimal_threshold = thresholds[optimal_idx]
                    
                    predicted_binary = (pin_edge_scores >= optimal_threshold).astype(int)

                    tp = ((predicted_binary == 1) & (all_labels == 1)).sum()
                    fp = ((predicted_binary == 1) & (all_labels == 0)).sum()
                    fn = ((predicted_binary == 0) & (all_labels == 1)).sum()
                    
                    precision_opt = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall_opt = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1_opt = 2 * precision_opt * recall_opt / (precision_opt + recall_opt + 1e-10) if (precision_opt + recall_opt) > 0 else 0

                    print(f"  Optimal threshold: {optimal_threshold:.3f}")
                    print(f"  F1 at optimal threshold: {f1_opt:.3f}")
                    print(f"  Precision/Recall at optimal: {precision_opt:.3f}/{recall_opt:.3f}")
                else:
                    f1_opt = 0
                    precision_opt = 0
                    recall_opt = 0
                    optimal_threshold = 0.5
                
                # Get top-k connection candidates
                k = min(5, len(pin_edge_scores))
                top_indices = np.argsort(pin_edge_scores)[-k:][::-1]
                
                print(f"  Top {k} connection candidates:")
                if representation == 'component_pin_net':
                    candidate_nodes = net_nodes
                else:
                    candidate_nodes = other_pins
                
                for rank, idx in enumerate(top_indices, 1):
                    candidate_node = candidate_nodes[idx]
                    is_correct = (candidate_node == (target_net if representation == 'component_pin_net' else target_node))
                    truth_label = "✓" if is_correct else "✗"
                    print(f"      {rank}. Node {candidate_node}: score={pin_edge_scores[idx]:.3f} {truth_label}")
                
            all_pin_results.append({
                'pin_idx': pin_idx,
                'pin_node': pin_node,
                'target_node': target_net if representation == 'component_pin_net' else target_node,
                'auc': auc_score,
                'f1': f1_opt,
                'precision': precision_opt,
                'recall': recall_opt,
                'top_predictions': [(candidate_nodes[i], pin_edge_scores[i]) for i in top_indices]
            })
            
            if all_pin_results:
                avg_pin_auc = np.mean([r['auc'] for r in all_pin_results])
                avg_pin_f1 = np.mean([r['f1'] for r in all_pin_results])

                results.append({
                    'target_node': target_component,
                    'actual_type': actual_type,
                    'predicted_type': predicted_type,
                    'correct_type': correct_type,
                    'num_pins': len(pin_nodes),
                    'pin_results': all_pin_results,
                    'avg_pin_auc': avg_pin_auc,
                    'avg_pin_f1': avg_pin_f1
                })
    
    # Summary
    print("*******************INFERENCE SUMMARY*******************")
    if results:
        correct_count = sum(1 for r in results if r['correct_type'])
        accuracy = correct_count / len(results) if results else 0
        print(f"Component type accuracy: {accuracy:.1%} ({correct_count}/{len(results)})")

    all_pins = [pin for r in results for pin in r['pin_results']]
    if all_pins:
        avg_auc = np.mean([r['auc'] for r in all_pins])
        avg_f1 = np.mean([r['f1'] for r in all_pins])
        avg_precision = np.mean([r['precision'] for r in all_pins])
        avg_recall = np.mean([r['recall'] for r in all_pins])
        
        print(f"\nLink Prediction Average (across {len(all_pins)} pins):")
        print(f"    AUC-ROC: {avg_auc:.3f} (primary metric)")
        print(f"    F1: {avg_f1:.3f}")
        print(f"    Precision: {avg_precision:.3f}")
        print(f"    Recall: {avg_recall:.3f}")

        top1_correct = 0
        top3_correct = 0
        
        for pin_result in all_pins:
            top_predictions = [node for node, _ in pin_result['top_predictions']]
            target_node = pin_result['target_node']
            
            if top_predictions and target_node == top_predictions[0]:
                top1_correct += 1
            if target_node in top_predictions[:min(3, len(top_predictions))]:
                top3_correct += 1
        
        top1_acc = top1_correct / len(all_pins) if all_pins else 0
        top3_acc = top3_correct / len(all_pins) if all_pins else 0
        
        print(f"\nTop-K Accuracy:")
        print(f"  Top-1 Accuracy: {top1_acc:.1%} ({top1_correct}/{len(all_pins)})")
        print(f"  Top-3 Accuracy: {top3_acc:.1%} ({top3_correct}/{len(all_pins)})")
    
    return results

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
        neg_sampling_ratio=2.0,
        max_pins=2
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
    model_path = Path("fegin_experiments") / "ltspice_demos_MultiTaskFEGIN__20260110211500_20260110_211500" / "models" / "best_multitask_model_iter_5.pth"
    results = demo_with_sample_circuit(model_path, device)
