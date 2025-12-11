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
    
    def forward(self, data, candidate_edges=None, task='both'):
        # task: 'classification', 'link_prediction', or 'both'
        # candidate_edges: tensor of shape [2, num_candidates] for link predictions
        # returns for classification: log_softmax predictions
        # for link_prediction: edge scores
        # for both: tuple of (classification_output, edge_scores)

        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Get node embeddings from shared encoder
        node_embeddings = self.encode(x, edge_index)
        
        if task == 'classification':
            # Pool to graph level for component classification
            graph_embeddings = global_mean_pool(node_embeddings, batch)
            logits = self.node_classifier(graph_embeddings)
            return F.log_softmax(logits, dim=1)
        
        elif task == 'link_prediction':
            # Predict edges for new component
            if candidate_edges is None:
                raise ValueError("candidate_edges must be provided for link prediction task")
            edge_scores = self.predict_edges(node_embeddings, candidate_edges, batch)
            return edge_scores
        
        else:  # both tasks
            # Component classification
            graph_embeddings = global_mean_pool(node_embeddings, batch)
            logits = self.node_classifier(graph_embeddings)
            class_output = F.log_softmax(logits, dim=1)
            
            # Link prediction
            if candidate_edges is None:
                raise ValueError("candidate_edges must be provided for link prediction task")
            edge_scores = self.predict_edges(node_embeddings, candidate_edges, batch)
            
            return class_output, edge_scores
    
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
        new_comp_repeated = new_component_embedding.unsqueeze(0).repeat(num_existing, 1)
        existing_embeddings = node_embeddings[existing_node_indices]
        edge_features = torch.cat([new_comp_repeated, existing_embeddings], dim=-1)
        # Predict edges
        edge_scores = self.edge_predictor(edge_features).squeeze(-1)
        
        return torch.sigmoid(edge_scores)

    def __repr__(self):
        return self.__class__.__name__
