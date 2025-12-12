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
        
        # Step 2: Convert masked graph to PyG Data
        data = convert_graph_to_pyg(G_masked, representation)
        if data is None:
            continue
        
        # Step 3: Create candidate edges
        node_mapping = {node: i for i, node in enumerate(G_masked.nodes())}
        candidate_edges = []
        
        for node in G_masked.nodes():
            idx = node_mapping[node]
            candidate_edges.append([idx, idx]) # self loops, edge prediction model basically predicts if a loop should be 'labeled' with 1 (= "node has a connection to the missing node") or 0 (="node does not have a connection to the missing node") 
        
        if candidate_edges:
            candidate_edges_tensor = torch.tensor(candidate_edges, dtype=torch.long).t()
            candidate_edges_list = [candidate_edges_tensor]
        else:
            candidate_edges_list = [torch.empty((2, 0), dtype=torch.long)]
        
        # Step 4: Prepare batch (single graph)
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
            
            # Edge predictions (connection scores)
            edge_scores_list = model(data_batch, candidate_edges=candidate_edges_list, task='link_prediction')
            
            if edge_scores_list and len(edge_scores_list[0]) > 0:
                edge_scores = edge_scores_list[0].cpu().numpy()
                
                # Get top-k connection candidates
                k = min(5, len(edge_scores))
                top_indices = np.argsort(edge_scores)[-k:][::-1]
                
                print(f"  Top {k} connection candidates:")
                for rank, idx in enumerate(top_indices, 1):
                    node_idx = list(node_mapping.keys())[list(node_mapping.values()).index(idx)]
                    node_type = G_masked.nodes[node_idx].get('type', 'unknown')
                    node_comp_type = G_masked.nodes[node_idx].get('comp_type', '')
                    score = edge_scores[idx]
                    
                    print(f"    {rank}. Node {node_idx} ({node_type}{'/'+node_comp_type if node_comp_type else ''}): "
                          f"score = {score:.3f}")
            
            # Check if prediction for component classification is correct
            correct_type = (predicted_type == actual_type)
            results.append({
                'target_node': target_component,
                'actual_type': actual_type,
                'predicted_type': predicted_type,
                'correct': correct_type
            })
    
    # Summary
    print("*******************INFERENCE SUMMARY*******************")
    correct_count = sum(1 for r in results if r['correct'])
    accuracy = correct_count / len(results) if results else 0
    print(f"Component type accuracy: {accuracy:.1%} ({correct_count}/{len(results)})")
    
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
        neg_sampling_ratio=1.0
    )
    
    model = load_trained_model(model_path, dataset, device)
    print(f"Loaded model from: {model_path}")
    
    # Load sample circuit graph
    graph_path = "../../graph_parsers/graphs_ltspice_demos/graphs_component_pin_net/LT1002_TA10_star_filtered.gpickle"
    
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
    model_path = Path("fegin_experiments") / "ltspice_demos_MultiTaskFEGIN__20251212110220_20251212_110220" / "models" / "best_multitask_model_iter_2.pth"
    demo_with_sample_circuit(model_path, device)

