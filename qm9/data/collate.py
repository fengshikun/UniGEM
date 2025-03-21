import torch


def batch_stack(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)


def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : Pytorch tensor
        Full Dataset


    Returns
    -------
    props : Pytorch tensor
        The dataset with  only the retained information.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0:
        return props
    elif props.shape[-1] == 53:
        return props # basic properties
    else:
        return props[:, to_keep, ...]


class PreprocessQM9:
    def __init__(self, load_charges=True):
        self.load_charges = load_charges

    def add_trick(self, trick):
        self.tricks.append(trick)

    def collate_fn(self, batch):
        """
        Collation function that collates datapoints into the batch format for cormorant

        Parameters
        ----------
        batch : list of datapoints
            The data to be collated.

        Returns
        -------
        batch : dict of Pytorch tensors
            The collated data.
        """
        new_batch = {}
        for prop in batch[0].keys():
            if prop not in ['edge_index', 'edge_attr']:
                new_batch[prop] = batch_stack([mol[prop] for mol in batch])
            if prop == 'edge_attr':
                new_batch[prop] = torch.cat([mol[prop] for mol in batch], dim=1)
            if prop == 'edge_index':
                # edge_index is a list of tensors, should be concatenated along the first dimension, the edge_index should add the offset of the previous molecule
                edge_index = []
                offset = 0
                for mol in batch:
                    edge_index.append(mol[prop] + offset)
                    offset += mol['num_atoms']
                new_batch[prop] = torch.cat(edge_index, dim=1)
                
        # batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}

        batch = new_batch
        to_keep = (batch['charges'].sum(0) > 0)

        for key, prop in batch.items():
            if key not in ['edge_index', 'edge_attr']:
                batch[key] = drop_zeros(prop, to_keep)
        
        # batch = {key: drop_zeros(prop, to_keep) for key, prop in batch.items()}

        atom_mask = batch['charges'] > 0
        batch['atom_mask'] = atom_mask

        #Obtain edges
        batch_size, n_nodes = atom_mask.size()
        edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

        #mask diagonal
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask

        #edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
        batch['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

        if self.load_charges:
            batch['charges'] = batch['charges'].unsqueeze(2)
        else:
            batch['charges'] = torch.zeros(0)
        return batch
