import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Data
from batch import Batch  # replace with custom Batch to handle subgraphs
import collections.abc as container_abcs
int_classes = int
string_classes = (str, bytes)

from torch_geometric.data import Batch

def multitask_dual_collate(batch):
    # batch item is dict with 'classification' (Data object -> component + all pin masked)
    # and 'pin_predictions' (list of Data pbjects -> one pin masked), 'candidate_edges' and 'edge_labels' as lists of tensors
    # 'pin_positions' list of pin indices

    class_batch = []
    all_pin_data = []         # Flattened list of all pin graphs
    all_cand_edges = []       # Flattened list of all candidate edges
    all_edge_labels = []      # Flattened list of all edge labels
    pin_positions = []        # Flattened list of all pin positions
    batch_indices = [] 
    
    for batch_idx, item in enumerate(batch):
        # Classification example
        class_batch.append(item['classification'])
        class_batch.append(item['pin_prediction'])
        # Add each pin prediction for this component
        num_pins = len(item['pin_predictions'])
        for i in range(num_pins):
            all_pin_data.append(item['pin_predictions'][i])
            all_cand_edges.append(item['candidate_edges'][i])
            all_edge_labels.append(item['edge_labels'][i])
            pin_positions.append(item['pin_positions'][i])
            batch_indices.append(batch_idx)

    
    # Batch examples
    class_batch = Batch.from_data_list(class_batch)
    pin_batch = Batch.from_data_list(all_pin_data)
    pin_positions = torch.tensor(pin_positions, dtype=torch.long)

    # Adjusted cand edge indices for batching
    adjusted_cand_edges = []
    node_counts = [data.num_nodes for data in all_pin_data]
    cumulative_offsets = [0]
    for count in node_counts:
        cumulative_offsets.append(cumulative_offsets[-1] + count)
    
    # TODO
    for i, (cand_edges, offset) in enumerate(zip(all_cand_edges, cumulative_offsets[:-1])):
        if cand_edges is not None and cand_edges.shape[1] > 0:
            adjusted = cand_edges.clone()
            adjusted[0, :] += offset
            adjusted[1, :] += offset
            adjusted_cand_edges.append(adjusted)
        else:
            adjusted_cand_edges.append(cand_edges)

    # Store batch indices to reconstruct which pins belong to which component
    batch_indices_tensor = torch.tensor(batch_indices, dtype=torch.long)
    
    return {
        'class_data': class_batch,
        'pin_data': pin_batch,
        'candidate_edges': adjusted_cand_edges,
        'edge_labels': all_edge_labels,
        'pin_positions': pin_positions,
        'batch_indices': batch_indices_tensor,
        'num_pins_per_example': torch.tensor([len(item['pin_predictions']) for item in batch], dtype=torch.long)
    }

def multitask_collate(batch):
    all_data = []
    all_cand_edges = []
    all_edge_labels = []

    # print(f"Debug collate - Batch size: {len(batch)}")

    for i, item in enumerate(batch):
        # print(f"  Graph {i}: num_nodes={item.num_nodes}, candidate_edges shape={item.candidate_edges.shape if item.candidate_edges is not None else 'None'}")
        # if item.candidate_edges is not None and item.candidate_edges.shape[1] > 0:
            # print(f"    First edge: {item.candidate_edges[:, 0].tolist()}, min idx: {item.candidate_edges.min().item()}, max idx: {item.candidate_edges.max().item()}")
        # item is a PyG Data object with attributes
        data_copy = item.clone()
        #cand_edges = data_copy.candidate_edges
        #edge_labels = data_copy.edge_labels
        cand_edges = getattr(data_copy, 'candidate_edges', None)
        edge_labels = getattr(data_copy, 'edge_labels', None)

        # Remove these attributes to avoid batching issues
        #del data_copy.candidate_edges
        #del data_copy.edge_labels
        if hasattr(data_copy, 'candidate_edges'):
            del data_copy.candidate_edges
        if hasattr(data_copy, 'edge_labels'):
            del data_copy.edge_labels

        all_data.append(data_copy)
        all_cand_edges.append(cand_edges)
        all_edge_labels.append(edge_labels)

    data_batch = Batch.from_data_list(all_data)
    # print(f"  Batched graph total nodes: {data_batch.num_nodes}")

    # Now we need to adjust candidate edge indices for batching offset
    # Calculate cumulative node counts for offset
    node_counts = [data.num_nodes for data in all_data]
    cumulative_offsets = [0]
    for count in node_counts:
        cumulative_offsets.append(cumulative_offsets[-1] + count)
    
    # Adjust candidate edges for each graph
    adjusted_candidate_edges = []
    for i, (cand_edges, offset) in enumerate(zip(all_cand_edges, cumulative_offsets[:-1])):
        if cand_edges is not None and cand_edges.shape[1] > 0:
            # Add offset to convert from local to batched indices
            adjusted = cand_edges.clone()
            adjusted[0, :] += offset  # Adjust source indices
            adjusted[1, :] += offset  # Adjust destination indices
            adjusted_candidate_edges.append(adjusted)
            # print(f"  Adjusted Graph {i}: offset={offset}, first edge after adjustment: {adjusted[:, 0].tolist()}")
        else:
            adjusted_candidate_edges.append(cand_edges)

    return data_batch, adjusted_candidate_edges, all_edge_labels


class DataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
        collate_fn (callable, optional): Custom collate function. If None,
            uses PyG default collate.
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],
                 collate_fn=None, **kwargs):
        if collate_fn is None:
            def collate(batch):
                elem = batch[0]
                if isinstance(elem, Data):
                    return Batch.from_data_list(batch, follow_batch)
                elif isinstance(elem, float):
                    return torch.tensor(batch, dtype=torch.float)
                elif isinstance(elem, int_classes):
                    return torch.tensor(batch)
                elif isinstance(elem, string_classes):
                    return batch
                elif isinstance(elem, container_abcs.Mapping):
                    return {key: collate([d[key] for d in batch]) for key in elem}
                elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
                    return type(elem)(*(collate(s) for s in zip(*batch)))
                elif isinstance(elem, container_abcs.Sequence):
                    return [collate(s) for s in zip(*batch)]

                raise TypeError('DataLoader found invalid type: {}'.format(
                    type(elem)))
            
            collate_fn = collate  # use default

        super(DataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=collate_fn, **kwargs)


class DataListLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a python list.

    .. note::

        This data loader should be used for multi-gpu support via
        :class:`torch_geometric.nn.DataParallel`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`False`)
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(DataListLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=lambda data_list: data_list, **kwargs)


class DenseDataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    .. note::

        To make use of this data loader, all graphs in the dataset needs to
        have the same shape for each its attributes.
        Therefore, this data loader should only be used when working with
        *dense* adjacency matrices.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`False`)
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        def dense_collate(data_list):
            batch = Batch()
            for key in data_list[0].keys:
                batch[key] = default_collate([d[key] for d in data_list])
            return batch

        super(DenseDataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=dense_collate, **kwargs)
