# Replica parser for BASELINE circuit representation with only component and net nodes (no pin nodes): component - net - component

import os
import networkx as nx
import pickle
from PySpice.Spice.Parser import SpiceParser
import numpy as np
from collections import Counter

COMPONENT_TYPES = ["R", "C", "V", "X"]
NODE_TYPES = ["component", "net"]

def clean_netlist_file(input_path, cleaned_path):
    with open(input_path, "r") as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        if any(param in line.lower() for param in ["rser=", "rpar=", "tol=", "temp=", "ic=", "tc="]):
            tokens = line.split()
            # keep element name, node connections, first numeric/model token
            keep = []
            for tok in tokens:
                if "=" in tok:  # stop before params
                    break
                keep.append(tok)
            cleaned_lines.append(" ".join(keep) + "\n")
        else:
            cleaned_lines.append(line)

    with open(cleaned_path, "w") as f:
        f.writelines(cleaned_lines)

def check_circuit_has_only_allowed_components(file_path):
    cleaned_path = file_path + ".clean"
    # Read netlist
    with open(cleaned_path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("*")]
    
    # ignore subcircuit definitions
    in_subckt = False
    filtered_lines = []
    for l in lines:
        l_upper = l.upper()
        if l_upper.startswith(".SUBCKT"):
            in_subckt = True
            continue
        elif l_upper.startswith(".ENDS"):
            in_subckt = False
            continue
        if not in_subckt:
            filtered_lines.append(l)
    
    # check for components
    allowed_prefixes = {'R', 'C', 'V', 'X', '.', 'K', '+'}  # . for directives, K for coupling, + for continuation
    
    for line in filtered_lines:
        if not line:
            continue
        first_char = line[0].upper()
        
        # Skip directives and special lines
        if first_char in {'.', 'K', '+'}:
            continue
        
        # Check if component type is allowed
        if first_char not in allowed_prefixes:
            print(f"Contains component type other than R, C, V, X: {first_char}")
            return False
    
    return True

def netlist_to_component_net_graph(file_path, use_edge_attributes=True):
    # clean netlist first
    cleaned_path = file_path + ".clean"
    clean_netlist_file(file_path, cleaned_path)
    
    # Check if circuit contains only allowed components
    if not check_circuit_has_only_allowed_components(file_path):
        print(f"Skipping {file_path}: contains disallowed component types")
        return None
    
    parser = SpiceParser(path=cleaned_path)
    try:
        circuit = parser.build_circuit()
    except Exception as e:
        print(f"Failed to parse {file_path}: {e}")
        return None
    
    G = nx.Graph()
    
    # Track component counts
    comp_counts = {ct: 0 for ct in COMPONENT_TYPES}
    
    # Process each component
    for element in circuit.element_names:
        comp_type = element[0].upper()
        
        # Skip if not in allowed component types
        if comp_type not in COMPONENT_TYPES:
            print(f"Skipping {element}: not in allowed component types")
            continue
        
        comp = circuit[element]
        nets = [str(net) for net in comp.nodes]
        
        # Add component node
        G.add_node(element, type="component", comp_type=comp_type, features=encode_component_features(comp_type))
        
        comp_counts[comp_type] += 1
        
        # Connect component to each of its nets
        for net in nets:
            # Add net node (will be added multiple times but thats fine due to same name)
            G.add_node(net, type="net", features=encode_net_features())
            
            # Add edge between component and net
            edge_attrs = { "kind": "component_net", "connection": "direct"}
            
            if use_edge_attributes:
                edge_attrs["weight"] = 1.0
            
            G.add_edge(element, net, **edge_attrs)
    
    # Clean up any isolated nodes
    isolated = list(nx.isolates(G))
    if isolated:
        print(f"Removing {len(isolated)} isolated nodes")
        G.remove_nodes_from(isolated)
    
    if G.number_of_nodes() == 0:
        print(f"Skipping {file_path}: empty graph")
        return None
    
    print(f"Component-net graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Components: {sum(comp_counts.values())} ({comp_counts})")
    
    return G

def encode_component_features(comp_type):
    return {
        "comp_type_idx": COMPONENT_TYPES.index(comp_type),
        "node_type_idx": NODE_TYPES.index("component")
    }

def encode_net_features():
    return {
        "node_type_idx": NODE_TYPES.index("net")
    }

def analyze_bipartite_structure(G):
    component_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "component"]
    net_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "net"]
    
    print("\nBipartite Analysis:")
    print(f"  Component nodes: {len(component_nodes)}")
    print(f"  Net nodes: {len(net_nodes)}")
    
    # Average degree
    comp_degrees = [G.degree(n) for n in component_nodes]
    net_degrees = [G.degree(n) for n in net_nodes]
    
    print(f"  Avg component degree: {np.mean(comp_degrees):.2f}")
    print(f"  Avg net degree: {np.mean(net_degrees):.2f}")
    
    # Net degree distribution
    degree_counts = Counter(net_degrees)
    print("  Net degree distribution:")
    for degree in sorted(degree_counts.keys()):
        print(f"    Degree {degree}: {degree_counts[degree]} nets")
    
    # Is graph truly bipartite?
    try:
        from networkx.algorithms import bipartite
        is_bipartite = bipartite.is_bipartite(G)
        print(f"  Is bipartite: {is_bipartite}")
    except:
        print("  (Could not verify bipartiteness)")

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    total_files = 0
    parsed_successfully = 0
    skipped_wrong_components = 0
    failed = 0
    
    for filename in os.listdir(input_folder):
        if filename.endswith((".cir", ".sp", ".net")):
            total_files += 1
            path = os.path.join(input_folder, filename)
            print(f"Processing {filename}")
            
            try:
                G = netlist_to_component_net_graph(path, use_edge_attributes=True)
                
                if G is None:
                    skipped_wrong_components += 1
                    continue
                
                # Save graph
                graph_filename = os.path.splitext(filename)[0] + "_component_net.gpickle"
                graph_path = os.path.join(output_folder, graph_filename)
                
                with open(graph_path, "wb") as f:
                    pickle.dump(G, f)
                
                print(f"  Saved to {graph_path}")
                parsed_successfully += 1
                
                analyze_bipartite_structure(G)
                
            except Exception as e:
                print(f"  Failed to parse {filename}: {e}")
                failed += 1
    
    print("PARSING SUMMARY\n")
    print(f"Total files:               {total_files}")
    print(f"Parsed successfully:       {parsed_successfully}")
    print(f"Skipped (wrong components): {skipped_wrong_components}")
    print(f"Failed (parsing errors):   {failed}")
    print(f"Success rate:              {parsed_successfully/total_files*100:.1f}%")

def remove_duplicate_graphs(folder):
    import hashlib
    
    print(f"Checking for duplicate graphs in {folder}\n")
    
    unique_hashes = {}
    duplicates = []
    
    def graph_signature(G):
        def serialize_features(fdict):
            if isinstance(fdict, dict):
                return tuple(sorted(fdict.items()))
            return fdict
        
        # Make node and edge lists deterministic
        node_data = sorted(
            (
                d.get("type"),
                d.get("comp_type"),
                serialize_features(d.get("features", {})),
            )
            for _, d in G.nodes(data=True)
        )
        
        edge_data = sorted(
            (tuple(sorted((u, v))), d.get("kind")) for u, v, d in G.edges(data=True)
        )
        
        # Hash everything
        m = hashlib.sha256()
        m.update(str(node_data).encode())
        m.update(str(edge_data).encode())
        return m.hexdigest()
    
    # Collect all graph files
    for fname in os.listdir(folder):
        if not fname.endswith(".gpickle"):
            continue
        
        path = os.path.join(folder, fname)
        
        with open(path, "rb") as f:
            G = pickle.load(f)
        
        sig = graph_signature(G)
        
        if sig in unique_hashes:
            orig = unique_hashes[sig]
            duplicates.append((fname, orig))
        else:
            unique_hashes[sig] = fname
    
    print(f"Found {len(duplicates)} duplicates out of {len(unique_hashes) + len(duplicates)} total graphs")
    
    # Remove duplicates
    for dup, orig in duplicates:
        os.remove(os.path.join(folder, dup))
        print(f"  Removed duplicate: {dup} (matched {orig})")
    
    print(f"\nFinal dataset: {len(unique_hashes)} unique graphs")

def analyze_dataset(folder):
    # Analyze component distribution in filtered dataset
    from collections import Counter
    
    print("DATASET ANALYSIS\n")
    
    comp_counter = Counter()
    graph_sizes = []
    
    for fname in os.listdir(folder):
        if not fname.endswith(".gpickle"):
            continue
            
        with open(os.path.join(folder, fname), 'rb') as f:
            G = pickle.load(f)
        
        graph_sizes.append(G.number_of_nodes())
        
        for node, attr in G.nodes(data=True):
            if attr.get("type") in ["component"]:
                comp_type = attr.get("comp_type")
                comp_counter[comp_type] += 1
    
    print("Component Type Distribution:")
    total = sum(comp_counter.values())
    for comp_type in COMPONENT_TYPES:
        count = comp_counter[comp_type]
        pct = count / total * 100 if total > 0 else 0
        print(f"  {comp_type:5s}: {count:5d} ({pct:5.1f}%)")
    
    print(f"Total components: {total}")
    print(f"Total graphs: {len(graph_sizes)}")
    print(f"Avg nodes per graph: {np.mean(graph_sizes):.1f}")
    print(f"Avg components per graph: {total / len(graph_sizes):.1f}")


if __name__ == "__main__":
    print("Netlist parser running...")
    
    input_folder = "netlists_amsnet"
    output_folder = "graphs_amsnet/graphs_component_net"
    process_folder(input_folder, output_folder)
    remove_duplicate_graphs(output_folder)
    analyze_dataset(output_folder)