from enum import Enum
from copy import deepcopy
from typing import NamedTuple, Dict, Union, TypeVar, List, Tuple, Iterable, Set, FrozenSet, OrderedDict, Callable, \
    Mapping, Optional

import sympy.combinatorics
from numpy.typing import NDArray

import numpy as np
from torch import Tensor

ODict = OrderedDict

Permutation = List[Tuple[str, ...]]
MappedPermutation = List[Tuple[int, ...]]

Permutations = List[Permutation]
MappedPermutations = List[MappedPermutation]

PermutationGroup = List[List[int]]
SymbolicPermutationGroup = sympy.combinatorics.PermutationGroup


class IndexDict():
    def __init__(self, init_dict, nEvent):
        self.dict_ = init_dict
        self.nEvent = nEvent

    def __getitem__(self, key):
        return self.dict_[key]

    def items(self):
        return self.dict_.items()

    def __len__(self):
        return self.nEvent

    def __iter__(self):
        return iter(self.dict_.keys())


class Particles:
    names: Tuple[str, ...]
    permutations: Permutations
    sources: Tuple[int, ...]

    def __init__(
            self,
            particles: Tuple[str, ...],
            permutations: Optional[Permutations] = None,
            sources: Optional[Tuple[int, ...]] = None
    ):
        self.names = particles
        self.permutations = permutations if permutations is not None else []
        self.sources = sources if sources is not None else tuple(-1 for _ in self.names)

    def __iter__(self) -> Iterable[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, item) -> str:
        return self.names[item]


# A 1-2 level dictionary structure for storing info about Feynmen diagrams.
Key = TypeVar("Key")
Value = TypeVar("Value")
NewValue = TypeVar("NewValue")
FeynmanDict = Dict[Key, Union[Value, Dict[Key, Value]]]

# To help differentiate that this object can only take event particles as keys.
EventDict = ODict

# To help differentiate that this object can only take product particles as keys.
ProductDict = ODict

# To help differentiate that this object can only take input sources particles as keys.
InputDict = ODict


# Similar to `tree_map` but only recurses over dictionaries so we can have lists of data at each node.
def feynman_map(function: Callable[[Value], NewValue], tree: FeynmanDict[Key, Value]) -> FeynmanDict[Key, NewValue]:
    return {
        key: feynman_map(function, value) if isinstance(value, dict) else function(value)
        for key, value in tree.items()
    }


def feynman_fill(
        tree: FeynmanDict[str, Value],
        event_particles: Particles,
        daughter_particles: Mapping[str, Particles],
        constructor: Callable[[], Value]
):
    tree = deepcopy(tree)

    if SpecialKey.Event not in tree:
        tree[SpecialKey.Event] = constructor()

    for particle in event_particles:
        if particle not in tree:
            tree[particle] = {}

        if SpecialKey.Particle not in tree[particle]:
            tree[particle][SpecialKey.Particle] = constructor()

        for daughter in daughter_particles[particle]:
            if daughter not in tree[particle]:
                tree[particle][daughter] = constructor()

    return tree


class Symmetries(NamedTuple):
    degree: int
    permutations: MappedPermutations


class RegressionInfo(NamedTuple):
    name: str
    type: str = "gaussian"


class FeatureInfo(NamedTuple):
    name: str
    normalize: bool
    log_scale: bool
    uniform: bool


ClassificationInfo = str


class SpecialKey(str, Enum):
    Mask = "MASK"
    Event = "EVENT"
    Inputs = "INPUTS"
    Targets = "TARGETS"
    Particle = "PARTICLE"
    Segmentations = "SEGMENTATIONS"
    Regressions = "REGRESSIONS"
    Permutations = "PERMUTATIONS"
    Classifications = "CLASSIFICATIONS"
    Embeddings = "EMBEDDINGS"
    ClassLabel = "CLASSLABEL"
    ParticleProperties = "PARTICLE_PROPERTIES"
    Generations = "GENERATIONS"
    Invisible = "Neutrinos"


class Source(NamedTuple):
    data: Tensor
    mask: Tensor


class SourceTuple(Tuple[Source, ...]):
    def __add__(self, other):
        result = []
        for index, source in enumerate(self):
            data, mask = source
            if isinstance(other, SourceTuple):
                data_o, mask_o = other[index]
                result.append(Source(data + data_o, mask))
            elif isinstance(other, list):
                result.append(Source(data + other[index], mask))
            else:
                result.append(Source(data + other, mask))
        return SourceTuple(tuple(result))

    def __mul__(self, other):
        result = []
        for index, source in enumerate(self):
            data, mask = source
            if isinstance(other, SourceTuple):
                data_o, mask_o = other[index]
                result.append(Source(data * data_o, mask))
            elif isinstance(other, list):
                result.append(Source(data * other[index], mask))
            else:
                result.append(Source(data * other, mask))

        return SourceTuple(tuple(result))

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __neg__(self):
        result = []
        for index, source in enumerate(self):
            data, mask = source
            result.append(Source(data * -1.0, mask))
        return SourceTuple(tuple(result))

    def __sub__(self, other):
        # Subtraction is defined as adding the negation
        return self.__add__(-other)

    def __rsub__(self, other):
        # Right subtraction is defined as other + (-self)
        return (-self).__add__(other)


class DistributionInfo(OrderedDict):
    def __add__(self, other):
        result = DistributionInfo()
        for key in self:
            if key in other:
                result[key] = Source(
                    data=self[key].data + other[key].data,
                    mask=self[key].mask
                )
            else:
                result[key] = self[key]
        for key in other:
            if key not in result:
                result[key] = other[key]
        return result

    def __sub__(self, other):
        result = DistributionInfo()
        for key in self:
            if key in other:
                result[key] = Source(
                    data=self[key].data - other[key].data,
                    mask=self[key].mask
                )
            else:
                result[key] = self[key]
        for key in other:
            if key not in result:
                result[key] = other[key]
        return result

    def __rsub__(self, other):
        result = DistributionInfo()
        for key in self:
            if key in other:
                result[key] = Source(
                    data=other[key].data - self[key].data,
                    mask=self[key].mask
                )
            else:
                result[key] = self[key]
        for key in other:
            if key not in result:
                result[key] = other[key]
        return result

    def __mul__(self, scalar):
        result = DistributionInfo()
        for key in self:
            result[key] = Source(
                data=self[key].data * scalar,
                mask=self[key].mask
            )
        return result

    def __rmul__(self, scalar):
        return self.__mul__(scalar)


class Statistics(NamedTuple):
    location: Tensor
    scale: Tensor


class InputType(str, Enum):
    Global = "GLOBAL"
    Relative = "RELATIVE"
    Sequential = "SEQUENTIAL"


class AssignmentTargets(NamedTuple):
    indices: Tensor
    mask: Tensor


class Batch(NamedTuple):
    sources: Tuple[Source, ...]
    num_vectors: int
    assignment_targets: Tuple[AssignmentTargets, ...]
    regression_targets: Dict[str, Tuple[Tensor, Tuple]]
    classification_targets: Dict[str, Tensor]
    num_sequential_vectors: Dict[str, Tensor]


class Outputs(NamedTuple):
    assignments: Dict[str, List[Tensor]]
    detections: Dict[str, List[Tensor]]
    vectors: Dict[str, Tensor]
    regressions: Dict[str, Tensor]
    classifications: Dict[str, Tensor]
    true_score: Dict[str, Tuple[Tensor, Tensor]]
    pred_score: Dict[str, Tuple[Tensor, Tensor]]


class Predictions(NamedTuple):
    assignments: Dict[str, List[NDArray[np.int64]]]
    detections: Dict[str, NDArray[np.float32]]
    regressions: Dict[str, NDArray[np.float32]]
    classifications: Dict[str, NDArray[np.int64]]


class Evaluation(NamedTuple):
    assignments: Dict[str, NDArray[np.int64]]
    assignment_probabilities: Dict[str, NDArray[np.float32]]
    detection_probabilities: Dict[str, NDArray[np.float32]]
    regressions: Dict[str, NDArray[np.float32]]
    classifications: Dict[str, NDArray[np.float32]]
    generations: Dict[str, NDArray[np.float32]]
    reference: Dict[str, NDArray[np.float32]]
    resonance: Dict[str, NDArray[np.float32]]
    resonance_target: Dict[str, NDArray[np.float32]]
    resonance_mask: Dict[str, NDArray[np.bool_]]
