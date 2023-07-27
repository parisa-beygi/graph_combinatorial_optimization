from torch.utils.data import Dataset
import torch
import os
import pickle

from problems.colouring.instance_loader import InstanceLoader
from problems.colouring.state_coloring import StateColoring
from utils.beam_search import beam_search
from IPython.core.debugger import set_trace
import numpy as np


class Coloring(object):
    NAME = 'coloring'

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
                torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
                pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return ColoringDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateColoring.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = Coloring.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class ColoringDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(ColoringDataset, self).__init__()

        self.data_set = []
        self.init_embedding_dim = 64
        if filename is not None:
            path = filename
            self.data_loader = InstanceLoader(path, num_samples)
            # train_loader.get_batches(16) returns: M, n_colors, MC, cn_exists, n_vertices, n_edges, f

            self.colors_initial_embeddings = [torch.FloatTensor(size, self.init_embedding_dim).uniform_(0, 1) for i in range(num_samples)]
            self.nodes_initial_embeddings = [torch.FloatTensor(size, self.init_embedding_dim).normal_(0, 1) for i in range(num_samples)]

            self.data = self.nodes_initial_embeddings
            set_trace()

            # assert os.path.splitext(filename)[1] == '.pkl'
            #
            # with open(filename, 'rb') as f:
            #     data = pickle.load(f)
            #     self.data = [torch.FloatTensor(row) for row in (data[offset:offset + num_samples])]
        # else:
            # Todo: handle by generating data

            # # Sample points randomly in [0, 1] square
            # self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
