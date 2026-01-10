import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class MultiTaskFEGIN(torch.nn.Module):

    def __init__(self, dataset, num_layers, hidden, emb_size, use_z=False, use_rd=False, 
                 lambda_node=1.0, lambda_edge=1.0, max_pins=2):
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
            torch.nn.Dropout(0.3),
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, dataset.num_classes)
        )

        # Component type embedding for link prediction
        self.comp_type_embedding = torch.nn.Embedding(
            dataset.num_classes,  # 4 component types: R, C, V, X
            num_layers * hidden   # Same dimension as node embeddings
        )

        self.max_pins = max_pins # 2 for R, C, V

        # Add pin position embeddings (for pin-specific predictions)
        self.pin_position_embedding = torch.nn.Embedding(
            max_pins,
            num_layers * hidden
        )

        self.edge_predictor = torch.nn.Sequential(
            Linear(3 * num_layers * hidden, hidden * 2),
            ReLU(),
            torch.nn.Dropout(0.4),
            Linear(hidden * 2, hidden),
            ReLU(),
            torch.nn.Dropout(0.3),
            Linear(hidden, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.edge_predictor:
                if isinstance(layer, Linear):
                    torch.nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)

        torch.nn.init.normal_(self.comp_type_embedding.weight, std=0.01)
        torch.nn.init.normal_(self.pin_position_embedding.weight, std=0.01)
    
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
    
    def forward(self, class_data, pin_data_list=None, candidate_edges_list=None, task='both', teacher_forcing=True, pin_position=None):
        # class_data: Graph with component+all pins masked (for classification)
        # pin_data_list: List of Graph with only one pin masked (for link prediction)
        # task: 'classification', 'link_prediction', or 'both'
        # candidate_edges_list: list of tensors of shape [2, num_candidates] for link predictions
        # returns for classification: log_softmax predictions
        # for link_prediction: list of edge scores
        # for both: tuple of (classification_output, list of edge_scores)

        
        if task == 'classification':
            x, edge_index, batch = class_data.x, class_data.edge_index, class_data.batch
        
            # Get node embeddings from shared encoder
            node_embeddings = self.encode(x, edge_index)
            # Pool to graph level for component classification
            graph_embeddings = global_mean_pool(node_embeddings, batch)
            logits = self.node_classifier(graph_embeddings)
            return F.log_softmax(logits, dim=1)
        

        #TODO: remove this option
        elif task == 'link_prediction':
            all_edge_scores = []

            for i, (pin_data, cand_edges) in enumerate(zip(pin_data_list, candidate_edges_list)):

                x_pin, edge_index_pin, batch_pin = pin_data.x, pin_data.edge_index, pin_data.batch
            
                # Get node embeddings from shared encoder
                node_emb_pin = self.encode(x_pin, edge_index_pin)

                # During training with teacher forcing: use true labels
                # During evaluation or inference: use predicted labels
                if teacher_forcing and self.training and hasattr(class_data, 'y'):
                    comp_type_indices = class_data.y.view(-1)[0] # same for all pins
                else:
                    comp_type_indices = class_output.max(1)[1][0]

                pin_pos = pin_position[i] if pin_position is not None else i

                # Process candidate edges
                edge_scores = self.process_candidate_edges(node_emb_pin, batch_pin, comp_type_indices, cand_edges, pin_pos)
                all_edge_scores.append(edge_scores)
            
            return all_edge_scores
        
        elif task == 'both':
            # 1. CLASSIFCATION
            x_class, edge_index_class, batch_class = class_data.x, class_data.edge_index, class_data.batch
            node_emb_class = self.encode(x_class, edge_index_class)
            graph_emb_class  = global_mean_pool(node_emb_class, batch_class)
            logits = self.node_classifier(graph_emb_class)
            class_output = F.log_softmax(logits, dim=1)

            # 2. MULTIPLE PIN PREDICTIONS
            all_edge_scores = []
            # Process each example's pins
            for example_idx, (example_pin_data, example_cand_edges) in enumerate(zip(pin_data_list, candidate_edges_list)):
                if not example_pin_data:  # Empty list
                    all_edge_scores.append([])
                    continue
                    
                example_edge_scores = []
                
                # Process each pin in this example
                for pin_idx, (pin_data, cand_edges) in enumerate(zip(example_pin_data, example_cand_edges)):
                    x_pin, edge_index_pin, batch_pin = pin_data.x, pin_data.edge_index, pin_data.batch
                    node_emb_pin = self.encode(x_pin, edge_index_pin)
                    
                    if teacher_forcing and self.training and hasattr(class_data, 'y'):
                        comp_type_idx = class_data.y.view(-1)[example_idx]
                    else:
                        comp_type_idx = class_output.max(1)[1][example_idx]
                    
                    if pin_position and example_idx < len(pin_position) and pin_idx < len(pin_position[example_idx]):
                        pin_pos = pin_position[example_idx][pin_idx]
                    else:
                        pin_pos = pin_idx
                    
                    edge_scores = self.process_candidate_edges_single_pin(
                        node_emb_pin, batch_pin, comp_type_idx, cand_edges, pin_pos
                    )
                    example_edge_scores.append(edge_scores)
            
                all_edge_scores.append(example_edge_scores)

            return class_output, all_edge_scores

    def process_candidate_edges_single_pin(self, node_embeddings, batch, comp_type_idx, candidate_edges, pin_pos):
        if candidate_edges is None or candidate_edges.shape[1] == 0:
            return torch.tensor([], device=node_embeddings.device)

        
        comp_type_emb = self.comp_type_embedding(torch.tensor([comp_type_idx], device=node_embeddings.device))
        # TODO: integrate pin embedding
        pin_emb = self.pin_position_embedding(torch.tensor([pin_pos], device=node_embeddings.device))
                
        src_indices = candidate_edges[0]
        dst_indices = candidate_edges[1]
        
        src_embs = node_embeddings[src_indices]
        dst_embs = node_embeddings[dst_indices]
        
        comp_type_embs = comp_type_emb.expand(src_embs.size(0), -1)
        
        edge_features = torch.cat([src_embs, dst_embs, comp_type_embs], dim=1)
        
        raw_scores = self.edge_predictor(edge_features).squeeze(-1)
        return torch.sigmoid(raw_scores)

    def __repr__(self):
        return self.__class__.__name__
