"""The island as an adaptive quadtree of 200 m cells: dense places split deep, empty sea stays one big node.

Node (L, a, b) covers the 2^L x 2^L cells (i, j) with (i - origin[0]) >> L == a and
(j - origin[1]) >> L == b. The root is (LEVELS, 0, 0), 2^13 cells (1 638.4 km) a side, enough for all of
Cuba. A node splits into its four children while it weighs more than SPLIT, where a building weighs 1 and a
road segment a quarter; the nodes that do not split are the leaves, and each leaf is one tile file. Python's
>> floors on negative numbers, so cells west and north of the Capitolio need no special case.
"""

import numpy as np

LEVELS = 13
SPLIT = 2000.0


def _keys(rel: np.ndarray, level: int) -> np.ndarray:
    """One int64 key per (a, b) pair of the cells' level-`level` nodes, for sorting and searching."""
    a, b = rel[:, 0] >> level, rel[:, 1] >> level
    return (a.astype(np.int64) << 20) | b.astype(np.int64)


def _sorted_unique(keys: np.ndarray) -> np.ndarray:
    """The distinct keys, sorted. np.unique hashes int64 keys and takes about fifteen times longer on the
    island's 2.8 million cells, in every process that loads the world."""
    keys = np.sort(keys)
    return keys[np.r_[True, keys[1:] != keys[:-1]]] if len(keys) else keys


class Tree:
    """The leaves of the quadtree, with their weights, and the cells that hold anything."""

    def __init__(self, origin: tuple[int, int], leaves: dict[tuple[int, int, int], float], cells=None):
        self.origin = (int(origin[0]), int(origin[1]))
        self.leaves = dict(leaves)
        self._inner = set()
        for L, a, b in self.leaves:
            for up in range(L + 1, LEVELS + 1):
                self._inner.add((up, a >> (up - L), b >> (up - L)))
        cells = np.zeros((0, 2), np.int64) if cells is None else np.asarray(cells, np.int64).reshape(-1, 2)
        rel = cells - np.array(self.origin)
        self._levels = []  # weighted nodes per level, as sorted keys
        for L in range(LEVELS + 1):  # each level from the one below: four times fewer keys to sort each time
            keys = _sorted_unique(_keys(rel, 0 if L == 0 else 1))
            self._levels.append(keys)
            rel = np.c_[keys >> 20, keys & ((1 << 20) - 1)]

    @classmethod
    def build(cls, weights: dict[tuple[int, int], float], split: float = SPLIT) -> "Tree":
        """Split from the root down while a node weighs more than `split`; level-0 nodes never split."""
        cells = np.array(list(weights), np.int64).reshape(-1, 2)
        w = np.array(list(weights.values()), float)
        origin = (int(cells[:, 0].min()) // 8 * 8, int(cells[:, 1].min()) // 8 * 8) if len(cells) else (0, 0)
        rel = cells - np.array(origin)
        if len(rel) and rel.max() >= 1 << LEVELS:
            raise ValueError("the cells do not fit in one quadtree root")
        leaves = {}

        def split_node(L, a, b, idx):
            total = float(w[idx].sum())
            if L == 0 or total <= split:
                leaves[(L, a, b)] = total
                return
            sub = (rel[idx, 0] >> (L - 1) & 1) * 2 + (rel[idx, 1] >> (L - 1) & 1)
            for k in range(4):
                part = idx[sub == k]
                if len(part):
                    split_node(L - 1, 2 * a + k // 2, 2 * b + k % 2, part)

        if len(rel):
            split_node(LEVELS, 0, 0, np.arange(len(rel)))
        return cls(origin, leaves, cells)

    def leaf_of(self, i: int, j: int) -> tuple[int, int, int] | None:
        """The leaf whose area holds cell (i, j), or None outside every leaf."""
        di, dj = int(i) - self.origin[0], int(j) - self.origin[1]
        for L in range(LEVELS + 1):
            key = (L, di >> L, dj >> L)
            if key in self.leaves:
                return key
        return None

    def exists(self, L: int, a: int, b: int) -> bool:
        """Whether node (L, a, b) holds anything: a leaf, an ancestor of one, or a weighted part of a leaf."""
        if (L, a, b) in self.leaves or (L, a, b) in self._inner:
            return True
        keys = self._levels[L]
        k = (np.int64(a) << 20) | np.int64(b)
        n = np.searchsorted(keys, k)
        return bool(n < len(keys) and keys[n] == k)

    def nodes(self) -> list[tuple[int, int, int]]:
        """Every leaf and every ancestor of one, coarsest first."""
        return sorted(set(self.leaves) | self._inner, key=lambda n: (-n[0], n[1], n[2]))

    def cells_in(self, L: int, a: int, b: int, cells: np.ndarray) -> np.ndarray:
        """The rows of an (N, 2) array of cells that lie in node (L, a, b)."""
        cells = np.asarray(cells, np.int64).reshape(-1, 2)
        rel = cells - np.array(self.origin)
        return cells[((rel[:, 0] >> L) == a) & ((rel[:, 1] >> L) == b)]
