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

def predict_single_component(model, original_graph, target_component, representation='component_pin_net', device='cuda'):

    print(f"\n{'='*80}")
    print(f"PREDICTING: {target_component}")
    print(f"{'='*80}")
    
    actual_type = original_graph.nodes[target_component].get('comp_type', 'Unknown')
    print(f"Actual component type: {actual_type}")
    
    # Create masked graph
    G_masked = original_graph.copy()
    removed_nodes = [target_component]
    
    # Get pin nodes
    if representation in ["component_pin", "component_pin_net"]:
        pin_nodes = [node for node in G_masked.neighbors(target_component) 
                    if G_masked.nodes[node].get('type') == 'pin']
        pin_nodes.sort(key=lambda n: original_graph.nodes[n].get('pin', ''))
        removed_nodes.extend(pin_nodes)
    else:
        pin_nodes = []
    
    G_masked.remove_nodes_from(removed_nodes)
    G_masked.remove_nodes_from(list(nx.isolates(G_masked)))
    
    if G_masked.number_of_nodes() < 2:
        print("  Not enough nodes after masking!")
        return None, []
    
    # Convert to PyG data
    data = convert_graph_to_pyg(G_masked, representation)
    if data is None:
        return None, []
    
    data_batch = Batch.from_data_list([data]).to(device)
    
    # TASK 1: COMPONENT CLASSIFICATION
    with torch.no_grad():
        class_output = model(data_batch, task='classification')
        predicted_class = class_output.max(1)[1].item()
        
        class_to_type = {0: 'R', 1: 'C', 2: 'V', 3: 'X'}
        predicted_type = class_to_type.get(predicted_class, 'Unknown')
        confidence = torch.exp(class_output[0, predicted_class]).item()
        
        correct = "✓" if predicted_type == actual_type else "✗"
        print(f"Predicted: {predicted_type} (confidence: {confidence:.3f}) {correct}")
    
    # TASK 2: PIN CONNECTION PREDICTION
    predicted_connections = []
    
    for pin_idx, pin_node in enumerate(pin_nodes):
        if pin_idx >= 2:
            break
        
        print(f"\n  Pin {pin_idx} ({pin_node}):")
        
        # Get ground truth
        if representation == 'component_pin_net':
            connected_nets = [n for n in original_graph.neighbors(pin_node) 
                            if original_graph.nodes[n].get('type') == 'net']
            if not connected_nets:
                continue
            target_node = connected_nets[0]
            candidate_nodes = [node for node in G_masked.nodes() 
                             if G_masked.nodes[node].get('type') == 'net']
            
        elif representation == 'component_pin':
            connected_pins = [n for n in original_graph.neighbors(pin_node)
                            if (original_graph.nodes[n].get('type') == 'pin' and 
                                n not in pin_nodes)]
            if not connected_pins:
                continue
            target_node = connected_pins[0]
            candidate_nodes = [node for node in G_masked.nodes() 
                             if (G_masked.nodes[node].get('type') == 'pin' and 
                                 node not in pin_nodes)]
        else:
            continue
        
        print(f"    True connection: {target_node}")
        
        # Create masked graph for this pin
        G_pin_masked = original_graph.copy()
        G_pin_masked.remove_node(pin_node)
        
        # Generate candidate edges
        node_mapping = {node: i for i, node in enumerate(G_pin_masked.nodes())}
        VIRTUAL_NODE_IDX = -1
        
        candidate_edges = []
        for node in candidate_nodes:
            if node in node_mapping:
                idx = node_mapping[node]
                candidate_edges.append([VIRTUAL_NODE_IDX, idx])
        
        if not candidate_edges:
            continue
        
        candidate_edges_tensor = torch.tensor(candidate_edges, dtype=torch.long).t()
        
        # Convert to PyG data
        pin_data = convert_graph_to_pyg(G_pin_masked, representation)
        if pin_data is None:
            continue
        
        pin_data.pin_position = torch.tensor([pin_idx], dtype=torch.long)
        pin_batch = Batch.from_data_list([pin_data]).to(device)
        
        # Predict connections
        with torch.no_grad():
            edge_scores = model(
                class_data=data_batch,
                pin_data_list=[[pin_batch]],
                candidate_edges_list=[[candidate_edges_tensor]],
                task='both',
                teacher_forcing=False,
                pin_position=[[pin_idx]]
            )[1]
            
            if edge_scores and edge_scores[0] and len(edge_scores[0]) > 0:
                pin_edge_scores = edge_scores[0][0].cpu().numpy()
            else:
                continue
        
        # Find top prediction
        top_idx = np.argmax(pin_edge_scores)
        predicted_node = candidate_nodes[top_idx]
        predicted_score = pin_edge_scores[top_idx]
        
        is_correct = (predicted_node == target_node)
        truth = "✓" if is_correct else "✗"
        
        print(f"    Predicted: {predicted_node} (score: {predicted_score:.3f}) {truth}")
        
        # Store prediction
        predicted_connections.append({
            'pin_idx': pin_idx,
            'pin_node': pin_node,
            'predicted_connection': predicted_node,
            'score': predicted_score,
            'true_connection': target_node,
            'correct': is_correct
        })
    
    return predicted_type, predicted_connections


def add_predicted_component_to_graph(G, component_node, predicted_type, predicted_connections, representation):
    
    print(f"\n  Adding predicted component {component_node} (type: {predicted_type}) to graph...")
    
    # Add component node
    G.add_node(component_node, type='component', comp_type=predicted_type)
    
    # Add pins and connections based on representation
    if representation in ['component_pin', 'component_pin_net']:
        for conn in predicted_connections:
            pin_node = conn['pin_node']
            predicted_connection = conn['predicted_connection']
            pin_idx = conn['pin_idx']
            
            if predicted_type == 'V':
                pin_label = 'pos' if pin_idx == 0 else 'neg'
            else:
                pin_label = str(pin_idx + 1)
            
            # Add pin node
            G.add_node(pin_node, type='pin', pin=pin_label)
            
            # Add component-to-pin edge
            G.add_edge(component_node, pin_node, kind='component_connection')
            
            if representation == 'component_pin_net':
                # Add pin-to-net edge
                G.add_edge(pin_node, predicted_connection, kind='net_connection')
            elif representation == 'component_pin':
                # Add pin-to-pin edge
                G.add_edge(pin_node, predicted_connection, kind='external')
    
    print(f"  ✓ Component added with {len(predicted_connections)} connections")
    
    return G


def predict_multiple_components(model, original_graph, target_components, representation='component_pin_net', device='cuda'):
    
    print(f"*****************MULTI-COMPONENT PREDICTION****************")
    print(f"Predicting {len(target_components)} components iteratively")
    print(f"***********************************************************")

    # All target components removed from graph
    working_graph = original_graph.copy()
    
    all_removed_nodes = []
    for comp_node in target_components:
        removed_nodes = [comp_node]
        
        if representation in ["component_pin", "component_pin_net"]:
            pin_nodes = [node for node in working_graph.neighbors(comp_node) 
                        if working_graph.nodes[node].get('type') == 'pin']
            removed_nodes.extend(pin_nodes)
        
        all_removed_nodes.extend(removed_nodes)
    
    working_graph.remove_nodes_from(all_removed_nodes)
    working_graph.remove_nodes_from(list(nx.isolates(working_graph)))
    
    print(f"Starting graph: {working_graph.number_of_nodes()} nodes, "
          f"{working_graph.number_of_edges()} edges")
    
    # Iteratively predict each component
    all_results = []
    
    for i, target_component in enumerate(target_components):
        print(f"****************ITERATION {i+1}/{len(target_components)}*****************")
        
        predicted_type, predicted_connections = predict_single_component(
            model, original_graph, target_component, representation, device
        )
        
        if predicted_type is None:
            print(f"✗ Failed to predict {target_component}")
            all_results.append({
                'component': target_component,
                'success': False
            })
            continue
        
        # Add prediction to working graph for next iteration    
        working_graph = add_predicted_component_to_graph(
            working_graph, target_component, predicted_type, 
            predicted_connections, representation
        )
        
        print(f"  Working graph now: {working_graph.number_of_nodes()} nodes, "
              f"{working_graph.number_of_edges()} edges")
        
        # Store results
        actual_type = original_graph.nodes[target_component].get('comp_type', 'Unknown')
        all_results.append({
            'component': target_component,
            'actual_type': actual_type,
            'predicted_type': predicted_type,
            'correct_type': predicted_type == actual_type,
            'connections': predicted_connections,
            'success': True
        })
    
    print("***********MULTI-COMPONENT PREDICTION SUMMARY************")
    
    successful = [r for r in all_results if r['success']]
    
    if successful:
        correct_types = sum(1 for r in successful if r.get('correct_type', False))
        print(f"\nComponent Type Accuracy: {correct_types}/{len(successful)} "
              f"({correct_types/len(successful)*100:.1f}%)")
        
        all_connections = [conn for r in successful 
                          for conn in r.get('connections', [])]
        if all_connections:
            correct_conns = sum(1 for c in all_connections if c['correct'])
            print(f"Connection Accuracy: {correct_conns}/{len(all_connections)} "
                  f"({correct_conns/len(all_connections)*100:.1f}%)")
        
        print(f"\nPrediction Details:")
        for i, result in enumerate(successful, 1):
            actual = result['actual_type']
            pred = result['predicted_type']
            correct = "✓" if result['correct_type'] else "✗"
            print(f"  {i}. {result['component']}: {actual} → {pred} {correct}")
    
    
    return all_results, working_graph


def convert_graph_to_pyg(G, representation):
    if G.number_of_nodes() == 0:
        return None
        
    node_features = []
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    
    node_features = get_node_features(G, representation)
    
    try:
        x = torch.tensor(np.array(node_features), dtype=torch.float)
    except Exception as e:
        print("ERROR: failed to convert node_features")
        return None

    if representation == "component_component":
        edges = [(node_mapping[u], node_mapping[v]) 
                for u, v in G.edges()]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        
    elif representation == "component_net":
        edges = [(node_mapping[u], node_mapping[v]) 
                for u, v in G.edges()]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        
    elif representation == "component_pin":
        edges = []
        edge_attrs = []
        for u, v, attr in G.edges(data=True):
            edges.append((node_mapping[u], node_mapping[v]))
            edge_kind = attr.get('kind', '')
            if edge_kind == 'internal':
                edge_attrs.append([1, 0])
            elif edge_kind == 'external':
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
        
    elif representation == "component_pin_net":
        edges = []
        edge_attrs = []
        for u, v, attr in G.edges(data=True):
            edges.append((node_mapping[u], node_mapping[v]))
            edge_kind = attr.get('kind', '')
            if edge_kind == 'component_connection':
                edge_attrs.append([1, 0])
            elif edge_kind == 'net_connection':
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

    degrees = dict(G.degree())
    node_features = []
    
    for node, attr in G.nodes(data=True):
        node_type = attr.get('type', '')
        comp_type = attr.get('comp_type', '')
        pin_type = attr.get('pin', '')
        
        if representation == "component_component":
            feat = np.zeros(6, dtype=np.float32)
            feat[0] = 1.0
            feat[1] = degrees[node] / 10.0
            if comp_type in ['R', 'C', 'V', 'X']:
                comp_idx = ['R', 'C', 'V', 'X'].index(comp_type)
                feat[2 + comp_idx] = 1.0
                
        elif representation == "component_net":
            feat = np.zeros(8, dtype=np.float32)
            if node_type == 'component':
                feat[0] = 1.0
                feat[1] = degrees[node] / 10.0
                if comp_type in ['R', 'C', 'V', 'X']:
                    comp_idx = ['R', 'C', 'V', 'X'].index(comp_type)
                    feat[2 + comp_idx] = 1.0
            else:  # net
                feat[6] = 1.0
                feat[7] = degrees[node] / 20.0
                
        elif representation in ["component_pin", "component_pin_net"]:
            feat = np.zeros(16, dtype=np.float32)
            if node_type == 'component':
                feat[0] = 1.0
                feat[1] = degrees[node] / 10.0
                if comp_type in ['R', 'C', 'V', 'X']:
                    comp_idx = ['R', 'C', 'V', 'X'].index(comp_type)
                    feat[2 + comp_idx] = 1.0
            elif node_type == 'pin':
                feat[6] = 1.0
                feat[7] = degrees[node] / 5.0
                if pin_type in ['1', '2', 'pos', 'neg', 'p']:
                    pin_idx = ['1', '2', 'pos', 'neg', 'p'].index(pin_type)
                    feat[8 + pin_idx] = 1.0
            elif node_type == 'net':
                feat[13] = 1.0
                feat[14] = degrees[node] / 20.0
        
        node_features.append(feat)
    
    return node_features

def demo_multiple_components(model_path, representation='component_pin_net', device='cuda'):
    print("******************DEMO: MULTI-COMPONENT ITERATIVE PREDICTION**********************")
    
    dataset = MultiTaskCircuitDataset(
        root="data/",
        name="ltspice_demos",
        representation=representation,
        h=2,
        max_nodes_per_hop=None,
        node_label="spd",
        use_rd=False,
        neg_sampling_ratio=2.0,
        max_pins=2
    )
    
    model = load_trained_model(model_path, dataset, device)
    
    graph_path = f"../../../graph_parsers/graphs_ltspice_demos/graphs_{representation}/LT1002_TA10_star_filtered.gpickle"
    
    if os.path.exists(graph_path):
        with open(graph_path, 'rb') as f:
            sample_graph = pickle.load(f)
        
        # Get component nodes
        component_nodes = [node for node, attr in sample_graph.nodes(data=True) 
                          if attr.get('type') == 'component']
        
        target_components = component_nodes[:2]
        
        print(f"\nWill predict: {target_components}")
        
        results, final_graph = predict_multiple_components(
            model, sample_graph, target_components, representation, device
        )
        
        return results, final_graph
    else:
        print(f"Graph file not found: {graph_path}")
        return None, None


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Path to trained model
    model_path = Path("fegin_experiments") / "ltspice_demos_MultiTaskFEGIN__20260103124550_20260103_124550" / "models" / "best_multitask_model_iter_6.pth"
    results = demo_multiple_components(model_path, 'component_pin_net', device)