import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from random import randint
import random
from random_shapes import random_shapes2

random_seed = 732
im_size = [64, 64]


class SyntheticDataset(torch.utils.data.Dataset):
    '''
    Implementation of a synthetic dataset by hierarchical diffusion.
    Args:
    :param int dim: dimension of the input sample
    :param int depth: depth of the tree; the root corresponds to the depth 0
    :param int :numberOfChildren: Number of children of each node in the tree
    :param float sigma_children: noise
    :param int param: integer by which :math:`\\sigma_children` is divided at each deeper level of the tree
    '''


    def __init__(self, dim, depth, numberOfChildren=2, sigma_children=1, param=2):
        self.dim = dim
        self.parent_idx = 0
        self.root = np.zeros(self.dim)
        self.depth = int(depth)
        self.sigma_children = sigma_children
        self.param = param
        self.numberOfChildren = int(numberOfChildren)

        self.total_nodes = int((1 - self.numberOfChildren ** self.depth) / (1 - self.numberOfChildren))
        self.adj = np.zeros((self.total_nodes, self.total_nodes), dtype=int)
        self.features = np.zeros((self.total_nodes, self.dim[0], self.dim[1]), dtype=float)
        self.parent_tracker = np.zeros((self.total_nodes, self.numberOfChildren), dtype=int)
        self.adj, self.features = self.bst()

        # Normalise data (0 mean, 1 std)
        #self.features -= np.mean(self.features, axis=0, keepdims=True)
        #self.features /= np.std(self.features, axis=0, keepdims=True)

    def __len__(self):
        return self.total_nodes

    def __getitem__(self):
        my_dict = {}
        for i in range(self.adj.shape[0]):
            con = np.where(self.adj[i, :] == 1)
            my_dict[i] = con[0].tolist()
        return my_dict, self.adj, self.features

    def __getadj__(self):
        return self.adj

    def __getfeature__(self):
        return self.features


    def get_children(self, parent_value, parent_idx, current_depth):
        decaying_factor = self.sigma_children / (self.param ** (current_depth))

        children = []
        last_assigned_node = self.parent_tracker[parent_idx - 1, 1]

        for i in range(self.numberOfChildren):
            child_idx = last_assigned_node + i + 1
            child_value = random_shapes2(im_size, max_shapes=1, multichannel=False,
                            input_data=parent_value,
                                         random_seed=np.random.randint(1000))
            children.append((child_value, child_idx))

            self.parent_tracker[parent_idx, i] = child_idx
            self.adj[parent_idx, child_idx] = 1
            self.features[child_idx] = np.squeeze(child_value[0])
        return children

    def bst(self):
        output = random_shapes2(im_size, max_shapes=1, multichannel=False, random_seed=random_seed)
        parent_value = np.squeeze(output[0])
        queue = [(output, 0, 0)]
        #output = random_shapes2(im_size, max_shapes=1, multichannel=False, random_seed=random_seed)
        #parent_value = np.squeeze(output[0])
        self.features[self.parent_idx] = parent_value
        while len(queue) > 0:
            current_node, current_idx, current_depth = queue.pop(0)
            if current_depth < self.depth - 1:
                children = self.get_children(current_node, current_idx, current_depth)
                for child in children:
                    queue.append((child[0], child[1], current_depth + 1))

        diag_one = np.zeros((self.total_nodes, self.total_nodes), dtype=int)
        np.fill_diagonal(diag_one, 1)
        self.adj = self.adj + self.adj.T
        return self.adj, self.features

    def graph_hierarchy(self):

        adjacancy = self.adj
        rows, cols = np.where(adjacancy == 1)
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.Graph()
        gr.add_edges_from(edges)
        pos = self.hierarchy_pos(gr, 1)
        nx.draw(gr, pos=pos, with_labels=True)
        plt.show()


    def hierarchy_pos(self, G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):


        '''
        From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
        Licensed under Creative Commons Attribution-Share Alike

        If the graph is a tree this will return the positions to plot this in a
        hierarchical layout.

        G: the graph (must be a tree)

        root: the root node of current branch
        - if the tree is directed and this is not given,
          the root will be found and used
        - if the tree is directed and this is given, then
          the positions will be just for the descendants of this node.
        - if the tree is undirected and not given,
          then a random choice will be used.

        width: horizontal space allocated for this branch - avoids overlap with other branches

        vert_gap: gap between levels of hierarchy

        vert_loc: vertical location of root

        xcenter: horizontal location of root
        '''
        if not nx.is_tree(G):
            raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

        if root is None:
            if isinstance(G, nx.DiGraph):
                root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
            else:
                root = random.choice(list(G.nodes))

        def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
            '''
            see hierarchy_pos docstring for most arguments

            pos: a dict saying where all nodes go if they have been assigned
            parent: parent of this branch. - only affects it if non-directed

            '''

            if pos is None:
                pos = {root: (xcenter, vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            children = list(G.neighbors(root))
            if not isinstance(G, nx.DiGraph) and parent is not None:
                children.remove(parent)
            if len(children) != 0:
                dx = width / len(children)
                nextx = xcenter - width / 2 - dx / 2
                for child in children:
                    nextx += dx
                    pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                         vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                         pos=pos, parent=root)
            return pos

        return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

    def show_graph_with_labels(self):
        adjacancy = self.adj
        position = {i: (self.features[i, 0], self.features[i, 1]) for i in range(self.features.shape[0])}
        labels = np.arange(adjacancy.shape[0])
        rows, cols = np.where(adjacancy == 1)
        edges = zip(rows.tolist(), cols.tolist())
        ax = plt.gca()
        for i in position:
            ax.text(position[i][0], position[i][1], s=str(labels[i]))

        gr = nx.Graph()
        gr.add_edges_from(edges)
        nx.draw_networkx_nodes(gr, position, node_color = 'r', node_size = 50, alpha = 1)

        for e in gr.edges:
            ax.annotate("",
                    xy=position[e[0]], xycoords='data',
                    xytext=position[e[1]], textcoords='data',
                    arrowprops=dict(arrowstyle="<-", color="0.5",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=rrr".replace('rrr', str(0.3 * randint(1, 3))
                                                                           ),
                                    ),
                    )



        plt.show()


if __name__ == '__main__':
    dim = [64, 64]
    adj_dict, adjacancy, feature = SyntheticDataset(dim=dim, depth=5).__getitem__()
    show_graph_with_labels(adjacancy, position=pos, labels=labels)

