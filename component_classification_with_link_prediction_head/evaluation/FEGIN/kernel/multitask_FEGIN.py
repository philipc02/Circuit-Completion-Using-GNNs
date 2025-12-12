import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class MultiTaskFEGIN(torch.nn.Module):
    """
    Multi-task FEGIN for simultaneous component classification and link prediction.
    
    Architecture:
    - Shared GNN encoder (GIN layers)
    - Component classification head (node-level)
    - Link prediction head (edge-level)
    """
    def __init__(self, dataset, num_layers, hidden, emb_size, use_z=False, use_rd=False, 
                 lambda_node=1.0, lambda_edge=1.0):
        super(MultiTaskFEGIN, self).__init__()
        self.use_z = use_z
        self.use_rd = use_rd
        self.lambda_node = lambda_node
        self.lambda_edge = lambda_edge
        self.num_layers = num_layers
        self.hidden = hidden
        
        input_dim = dataset.num_features
        
        # Shared GNN Encoder
        self.conv1 = GINConv(
            Sequential(
                Linear(input_dim, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ),
            train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ),
                    train_eps=True))
        
        # Component classification head
        self.node_classifier = torch.nn.Sequential(
            Linear(num_layers * hidden, hidden * 2),
            ReLU(),
            torch.nn.Dropout(0.5),
            Linear(hidden * 2, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, dataset.num_classes)
        )

        # Component type embedding for link prediction
        self.comp_type_embedding = torch.nn.Embedding(
            dataset.num_classes,  # 4 component types: R, C, V, X
            num_layers * hidden   # Same dimension as node embeddings
        )
        
        # Link prediction head
        # Takes concatenated node embeddings and predicts edge probability
        self.edge_predictor = torch.nn.Sequential(
            Linear(2 * num_layers * hidden, hidden * 2),
            ReLU(),
            torch.nn.Dropout(0.3),
            Linear(hidden * 2, hidden),
            ReLU(),
            Linear(hidden, 1)
        )
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        
        for layer in self.node_classifier:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        for layer in self.edge_predictor:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs.append(x)
        
        # Concatenate all layer outputs
        node_embeddings = torch.cat(xs, dim=-1)
        return node_embeddings
    
    def forward(self, data, candidate_edges=None, task='both', teacher_forcing=True):
        # task: 'classification', 'link_prediction', or 'both'
        # candidate_edges: tensors of shape [2, num_candidates] for link predictions
        # returns for classification: log_softmax predictions
        # for link_prediction: list of edge scores
        # for both: tuple of (classification_output, list of edge_scores)

        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Get node embeddings from shared encoder
        node_embeddings = self.encode(x, edge_index)
        
        if task == 'classification':
            # Pool to graph level for component classification
            graph_embeddings = global_mean_pool(node_embeddings, batch)
            logits = self.node_classifier(graph_embeddings)
            return F.log_softmax(logits, dim=1)
        
        if task == 'link_prediction' or task == 'both':
            edge_scores_list = []
            # Component classification
            graph_embeddings = global_mean_pool(node_embeddings, batch)
            logits = self.node_classifier(graph_embeddings)
            class_output = F.log_softmax(logits, dim=1)

            # During training with teacher forcing: use true labels
            # During evaluation or inference: use predicted labels
            if teacher_forcing and self.training and hasattr(data, 'y'):
                comp_type_indices = data.y.view(-1)
            else:
                comp_type_indices = class_output.max(1)[1]
            
            # Link prediction - process each graph in batch separately
            if candidate_edges is not None:
                # Get batch indices for each node
                batch_size = batch.max().item() + 1
                batch_node_indices = []
                
                # Get node indices for each graph in batch
                for i in range(batch_size):
                    batch_node_indices.append((batch == i).nonzero(as_tuple=True)[0])
                
                # Process candidate edges for each graph
                for i, (candidate_edges_i, node_indices) in enumerate(zip(candidate_edges, batch_node_indices)):


                    if candidate_edges_i is not None and candidate_edges_i.shape[1] > 0:
                        # Adjust indices to be within this graphs node indices
                        # First get the node embeddings for this graph
                        graph_node_embeddings = node_embeddings[node_indices]
                        # Get component type embedding for this graph
                        comp_type_idx = comp_type_indices[i]
                        comp_type_emb = self.comp_type_embedding(comp_type_idx)

                        scores  = []
                        
                        for j in range(candidate_edges_i.shape[1]):
                            src = candidate_edges_i[0, j].item()
                            dst = candidate_edges_i[1, j].item()
                            
                            # src should be -1 (virtual node), dst is the candidate node
                            if src == -1 and 0 <= dst < len(graph_node_embeddings):
                                dst_emb = graph_node_embeddings[dst]
                                
                                edge_feature = torch.cat([comp_type_emb, dst_emb], dim=0)
                                edge_feature = edge_feature.unsqueeze(0)  # Add batch dimension
                                score = self.edge_predictor(edge_feature).squeeze()
                                scores.append(torch.sigmoid(score))
                            else:
                                # Skip invalid edges
                                continue

                        if scores:
                            edge_scores_list.append(torch.stack(scores))
                        else:
                            edge_scores_list.append(torch.tensor([], device=x.device))
                    else:
                        edge_scores_list.append(torch.tensor([], device=x.device))
                
                if task == 'link_prediction':
                    return edge_scores_list
                else:
                    return class_output, edge_scores_list
            else:
                if task == 'link_prediction':
                    return [torch.tensor([], device=x.device) for _ in range(batch_size)]
                else:
                    return class_output, [torch.tensor([], device=x.device) for _ in range(batch_size)]
    
    def predict_edges(self, node_embeddings, candidate_edges, batch):
        # predict edge scores for candidate edges.
        # node_embeddings: [num_nodes, emb_size]
        # candidate_edges: [2, num_candidates]
        # batch: [num_nodes] batch assignment
        # returns edge_scores: Probability scores for each candidate edge [num_candidates]

        
        # Get embeddings for source and destination nodes
        src_embeddings = node_embeddings[candidate_edges[0]]  # [num_candidates, emb_size]
        dst_embeddings = node_embeddings[candidate_edges[1]]  # [num_candidates, emb_size]
        # Concatenate source and destination embeddings -> edge feature
        edge_features = torch.cat([src_embeddings, dst_embeddings], dim=1)
        # Predict edge probability
        edge_scores = self.edge_predictor(edge_features).squeeze(-1)
        
        return torch.sigmoid(edge_scores)
    
    def decode_edges_for_new_component(self, node_embeddings, new_component_embedding, 
                                       existing_node_indices):
        # Predict which existing nodes new component should connect to -> during inference/test time
        # return edge_scores: Probability of connection to each existing node

        num_existing = len(existing_node_indices)
        
        # Repeat new component embedding for each potential connection
        # new_comp_embedding: [128] -> unsqueeze to [1, 128] -> repeat to [3, 128] for emb_size = 128
        new_comp_repeated = new_component_embedding.unsqueeze(0).repeat(num_existing, 1)
        # Result: [[comp_emb], [comp_emb], [comp_emb]]  (3 copies)
        existing_embeddings = node_embeddings[existing_node_indices]
        # Result: [[node_A_emb], [node_B_emb], [node_C_emb]]
        # Each edge feature = [comp_emb || node_emb]
        edge_features = torch.cat([new_comp_repeated, existing_embeddings], dim=-1)
        # Result shape: [3, 256] (128 + 128)
        # Predict edges
        edge_scores = self.edge_predictor(edge_features).squeeze(-1)
        # Result: [3] with raw scores
        
        # Result: [3] with values 0-1 (probability of connection)
        return torch.sigmoid(edge_scores)

    def __repr__(self):
        return self.__class__.__name__
