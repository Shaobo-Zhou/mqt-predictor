"""Helper functions of the reinforcement learning compilation predictor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import requests
import networkx as nx
from bqskit import MachineModel
from pytket.architecture import Architecture
from pytket.circuit import Circuit, Node, Qubit
from pytket.passes import (
    CliffordSimp,
    FullPeepholeOptimise,
    PeepholeOptimise2Q,
    RemoveRedundancies,
    RoutingPass,
)
from pytket.placement import place_with_map
from qiskit import QuantumCircuit
from qiskit.circuit.equivalence_library import StandardEquivalenceLibrary
from qiskit.circuit.library import XGate, YGate, ZGate, CXGate, CYGate, CZGate, HGate, TGate, TdgGate, SdgGate, SGate, SwapGate
from qiskit.transpiler import CouplingMap, Layout, PassManager, TranspileLayout
from qiskit.transpiler.passes import (
    ApplyLayout,
    BasicSwap,
    BasisTranslator,
    CheckMap,
    Collect2qBlocks,
    CollectCliffords,
    CommutativeCancellation,
    CommutativeInverseCancellation,
    ConsolidateBlocks,
    DenseLayout,
    Depth,
    EnlargeWithAncilla,
    FixedPoint,
    FullAncillaAllocation,
    GatesInBasis,
    InverseCancellation,
    LookaheadSwap,
    MinimumPoint,
    Optimize1qGatesDecomposition,
    OptimizeCliffords,
    RemoveDiagonalGatesBeforeMeasure,
    SabreLayout,
    SabreSwap,
    Size,
    TrivialLayout,
    
    UnitarySynthesis,
    VF2Layout,
    VF2PostLayout,
)

from qiskit_ibm_transpiler.ai.routing import AIRouting
from qiskit_ibm_transpiler.ai.collection import CollectLinearFunctions, CollectPermutations
from qiskit_ibm_transpiler.ai.synthesis import (
    AICliffordSynthesis,
    AILinearFunctionSynthesis,
    AIPermutationSynthesis,
)

from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
from sb3_contrib import MaskablePPO
from tqdm import tqdm

from mqt.bench.utils import calc_supermarq_features
from mqt.predictor import reward, rl

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from qiskit.providers.models import BackendProperties

    from mqt.bench.devices import Device


import operator
import os
import zipfile
from importlib import resources

from bqskit import compile as bqskit_compile
from bqskit.ir import gates
from qiskit import QuantumRegister
from qiskit.passmanager import ConditionalController
from qiskit.transpiler.preset_passmanagers import common
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2, FakeMontrealV2, FakeQuitoV2, FakeWashingtonV2

logger = logging.getLogger("mqt-predictor")


def qcompile(
    qc: QuantumCircuit | str,
    figure_of_merit: reward.figure_of_merit | None = "expected_fidelity",
    device_name: str | None = "ibm_washington",
    predictor_singleton: rl.Predictor | None = None,
) -> tuple[QuantumCircuit, list[str]]:
    """Compiles a given quantum circuit to a device optimizing for the given figure of merit.

    Arguments:
        qc: The quantum circuit to be compiled. If a string is given, it is assumed to be a path to a qasm file.
        figure_of_merit: The figure of merit to be used for compilation. Defaults to "expected_fidelity".
        device_name: The name of the device to compile to. Defaults to "ibm_washington".
        predictor_singleton: A predictor object that is used for compilation to reduce compilation time when compiling multiple quantum circuits. If None, a new predictor object is created. Defaults to None.

    Returns:
        A tuple containing the compiled quantum circuit and the compilation information. If compilation fails, False is returned.
    """
    if predictor_singleton is None:
        if figure_of_merit is None:
            msg = "figure_of_merit must not be None if predictor_singleton is None."
            raise ValueError(msg)
        if device_name is None:
            msg = "device_name must not be None if predictor_singleton is None."
            raise ValueError(msg)
        predictor = rl.Predictor(figure_of_merit=figure_of_merit, device_name=device_name)
    else:
        predictor = predictor_singleton

    return predictor.compile_as_predicted(qc)


def get_actions_opt() -> list[dict[str, Any]]:
    """Returns a list of dictionaries containing information about the optimization passes that are available."""
    return [
        {
            "name": "Optimize1qGatesDecomposition",
            "transpile_pass": lambda native_gate: [Optimize1qGatesDecomposition(basis=native_gate)],
            "origin": "qiskit",
        },
        {
            "name": "CommutativeCancellation",
            "transpile_pass": [CommutativeCancellation()],
            "origin": "qiskit",
        },
        {
            "name": "CommutativeInverseCancellation",
            "transpile_pass": [CommutativeInverseCancellation()],
            "origin": "qiskit",
        },
        {
            "name": "RemoveDiagonalGatesBeforeMeasure",
            "transpile_pass": [RemoveDiagonalGatesBeforeMeasure()],
            "origin": "qiskit",
        },
        {
            "name": "InverseCancellation",
            "transpile_pass": [InverseCancellation([
                XGate(), YGate(), ZGate(),         
                HGate(),                      
                CXGate(), CYGate(), CZGate(),    
                SwapGate(),                   
                (TGate(), TdgGate()), (SGate(), SdgGate())
            ])],
            "origin": "qiskit",
        },
        {
            "name": "OptimizeCliffords",
            "transpile_pass": [CollectCliffords(), OptimizeCliffords()],
            "origin": "qiskit",
        },
        # {
        #     "name": "AIClifford",
        #     "transpile_pass": lambda coupling_map: [
        #         CollectCliffords(),
        #         AICliffordSynthesis(coupling_map=coupling_map)
        #     ],
        #     "origin": "qiskit_ai"
        # },
        # Very restricted usecase and underperforms AIClifford
        # {   "name": "AILinearFunction", 
        #     "transpile_pass": [CollectLinearFunctions(), AILinearFunctionSynthesis()],
        #     "origin": "qiskit_ai"
        # },

        # Very restricted usecase
        # {   "name": "AIPermutation", 
        #     "transpile_pass": [CollectPermutations(), AIPermutationSynthesis()],
        #     "origin": "qiskit_ai"
        # },
        {
            "name": "Opt2qBlocks",
            "transpile_pass": lambda native_gate: [Collect2qBlocks(), ConsolidateBlocks(basis_gates=native_gate)],
            "origin": "qiskit",
        },
        
        {
            "name": "PeepholeOptimise2Q",
            "transpile_pass": [PeepholeOptimise2Q()],
            "origin": "tket",
        },
        {
            "name": "CliffordSimp",
            "transpile_pass": [CliffordSimp()],
            "origin": "tket",
        },
        {
            "name": "FullPeepholeOptimiseCX",
            "transpile_pass": [FullPeepholeOptimise()],
            "origin": "tket",
        },
        {
            "name": "RemoveRedundancies",
            "transpile_pass": [RemoveRedundancies()],
            "origin": "tket",
        },
        # {
        #     "name": "QiskitO3",
        #     "transpile_pass": lambda native_gate, coupling_map: [
        #         Collect2qBlocks(),
        #         ConsolidateBlocks(basis_gates=native_gate),
        #         UnitarySynthesis(basis_gates=native_gate, coupling_map=coupling_map),
        #         Optimize1qGatesDecomposition(basis=native_gate),
        #         CommutativeCancellation(basis_gates=native_gate),
        #         GatesInBasis(native_gate),
        #         ConditionalController(
        #             common.generate_translation_passmanager(
        #                 target=None, basis_gates=native_gate, coupling_map=coupling_map
        #             ).to_flow_controller(),
        #             condition=lambda property_set: not property_set["all_gates_in_basis"],
        #         ),
        #         Depth(recurse=True),
        #         FixedPoint("depth"),
        #         Size(recurse=True),
        #         FixedPoint("size"),
        #         MinimumPoint(["depth", "size"], "optimization_loop"),
        #     ],
        #     "origin": "qiskit",
        #     "do_while": lambda property_set: (not property_set["optimization_loop_minimum_point"]),
        # },
        # {
        #     "name": "BQSKitO2",
        #     "transpile_pass": lambda circuit: bqskit_compile(
        #         circuit,
        #         optimization_level=1 if os.getenv("GITHUB_ACTIONS") == "true" else 2,
        #         #synthesis_epsilon=1e-1 if os.getenv("GITHUB_ACTIONS") == "true" else 1e-8,
        #         synthesis_epsilon=1e-1 if os.getenv("GITHUB_ACTIONS") == "true" else 1e-4,
        #         max_synthesis_size=2 if os.getenv("GITHUB_ACTIONS") == "true" else 3,
        #         seed=10,
        #     ), 
        #     "origin": "bqskit",     
        # },
        # {
        #     "name": "AICliffordSynthesis",
        #     "transpile_pass": lambda device: [AICliffordSynthesis(coupling_map=device.coupling_map)],
        #     "origin": "qiskit_ai",
        # },
        # {
        #     "name": "AILinearFunctionSynthesis",
        #     "transpile_pass": lambda device: [AILinearFunctionSynthesis(coupling_map=device.coupling_map)],
        #     "origin": "qiskit_ai",
        # },
        # {
        #     "name": "AIPermutationSynthesis",
        #     "transpile_pass": lambda device: [AIPermutationSynthesis(coupling_map=device.coupling_map)],
        #     "origin": "qiskit_ai",
        # },
    ]


def get_actions_final_optimization() -> list[dict[str, Any]]:
    """Returns a list of dictionaries containing information about the optimization passes that are available."""
    return [
        {
            "name": "VF2PostLayout",
            "transpile_pass": lambda device: VF2PostLayout(
                target=get_ibm_backend_properties_by_device_name(device.name), time_limit= 10
            ),
            "origin": "qiskit",
        }
    ]


def get_actions_layout() -> list[dict[str, Any]]:
    """Returns a list of dictionaries containing information about the layout passes that are available."""
    return [
        {
            "name": "TrivialLayout",
            "transpile_pass": lambda device: [
                TrivialLayout(coupling_map=CouplingMap(device.coupling_map)),
                FullAncillaAllocation(coupling_map=CouplingMap(device.coupling_map)),
                EnlargeWithAncilla(),
                ApplyLayout(),
            ],
            "origin": "qiskit",
        },
        {
            "name": "DenseLayout",
            "transpile_pass": lambda device: [
                DenseLayout(coupling_map=CouplingMap(device.coupling_map)),
                FullAncillaAllocation(coupling_map=CouplingMap(device.coupling_map)),
                EnlargeWithAncilla(),
                ApplyLayout(),
            ],
            "origin": "qiskit",
        },
        {
            "name": "VF2Layout",
            "transpile_pass": lambda device: [
                VF2Layout(
                    coupling_map=CouplingMap(device.coupling_map),
                    target=get_ibm_backend_properties_by_device_name(device.name),
                ),
                ConditionalController(
                    [
                        FullAncillaAllocation(coupling_map=CouplingMap(device.coupling_map)),
                        EnlargeWithAncilla(),
                        ApplyLayout(),
                    ],
                    condition=lambda property_set: property_set["VF2Layout_stop_reason"]
                    == VF2LayoutStopReason.SOLUTION_FOUND,
                ),
            ],
            "origin": "qiskit",
        },

        {
            "name": "SabreLayout",
            "transpile_pass": lambda device: [
                SabreLayout(
                    coupling_map=CouplingMap(device.coupling_map),
                    skip_routing=True,        
                ),
                FullAncillaAllocation(coupling_map=CouplingMap(device.coupling_map)),
                EnlargeWithAncilla(),
                ApplyLayout(),
            ],
            "origin": "qiskit",
        },
    ]


def get_actions_routing() -> list[dict[str, Any]]:
    """Returns a list of dictionaries containing information about the routing passes that are available."""
    return [
        {
            "name": "BasicSwap",
            "transpile_pass": lambda device: [BasicSwap(coupling_map=CouplingMap(device.coupling_map))],
            "origin": "qiskit",
        },

        # {
        #     "name": "LookaheadSwap",
        #     "transpile_pass": lambda device: [LookaheadSwap(coupling_map=CouplingMap(device.coupling_map))],
        #     "origin": "qiskit",
        # }

        {
            "name": "SabreSwap",
            "transpile_pass": lambda device: [SabreSwap(coupling_map=CouplingMap(device.coupling_map), 
                                                        heuristic="decay")],
            "origin": "qiskit",
        },

        {
            "name": "RoutingPass",
            "transpile_pass": lambda device: [
                PreProcessTKETRoutingAfterQiskitLayout(),
                RoutingPass(Architecture(device.coupling_map)),
            ],
            "origin": "tket",
        },
        {
            "name": "AIRouting",
            "transpile_pass": lambda device: [AIRouting(
                coupling_map=device.coupling_map,
                optimization_level=2,
                layout_mode="optimize",
                local_mode=True
            )],
            "origin": "qiskit_ai",
        },

    ]


def get_actions_mapping() -> list[dict[str, Any]]:
    """Returns a list of dictionaries containing information about the mapping passes that are available."""
    return [
        {
            "name": "SabreMapping",
            "transpile_pass": lambda device: [
                SabreLayout(coupling_map=CouplingMap(device.coupling_map), skip_routing=False),
            ],
            "origin": "qiskit",
        },
        # {
        #     "name": "BQSKitMapping",
        #     "transpile_pass": lambda device: lambda bqskit_circuit: bqskit_compile(
        #         bqskit_circuit,
        #         model=MachineModel(
        #             num_qudits=device.num_qubits,
        #             gate_set=get_bqskit_native_gates(device),
        #             coupling_graph=[(elem[0], elem[1]) for elem in device.coupling_map],
        #         ),
        #         with_mapping=True,
        #         optimization_level=1 if os.getenv("GITHUB_ACTIONS") == "true" else 2,
        #         #synthesis_epsilon=1e-1 if os.getenv("GITHUB_ACTIONS") == "true" else 1e-8,
        #         synthesis_epsilon=1e-1 if os.getenv("GITHUB_ACTIONS") == "true" else 1e-4,
        #         max_synthesis_size=2 if os.getenv("GITHUB_ACTIONS") == "true" else 3,
        #         seed=10,
        #     ),
        #     "origin": "bqskit",
        # },
    ]


def get_actions_synthesis() -> list[dict[str, Any]]:
    """Returns a list of dictionaries containing information about the synthesis passes that are available."""
    return [
        {
            "name": "BasisTranslator",
            "transpile_pass": lambda device: [
                BasisTranslator(StandardEquivalenceLibrary, target_basis=device.basis_gates)
            ],
            "origin": "qiskit",
        },
        # {
        #     "name": "BQSKitSynthesis",
        #     "transpile_pass": lambda device: lambda bqskit_circuit: bqskit_compile(
        #         bqskit_circuit,
        #         model=MachineModel(bqskit_circuit.num_qudits, gate_set=get_bqskit_native_gates(device)),
        #         optimization_level=1 if os.getenv("GITHUB_ACTIONS") == "true" else 2,
        #         #synthesis_epsilon=1e-1 if os.getenv("GITHUB_ACTIONS") == "true" else 1e-8,
        #         synthesis_epsilon=1e-1 if os.getenv("GITHUB_ACTIONS") == "true" else 1e-4,
        #         max_synthesis_size=2 if os.getenv("GITHUB_ACTIONS") == "true" else 3,
        #         seed=10,
        #     ),
        #     "origin": "bqskit",
        # },
        
    ]


def get_action_terminate() -> dict[str, Any]:
    """Returns a dictionary containing information about the terminate pass that is available."""
    return {"name": "terminate"}


def get_state_sample(max_qubits: int | None = None, rng: int | None = None) -> tuple[QuantumCircuit, str]:
    """Returns a random quantum circuit from the training circuits folder.

    Arguments:
        max_qubits: The maximum number of qubits the returned quantum circuit may have. If no limit is set, it defaults to None.

    Returns:
        A tuple containing the random quantum circuit and the path to the file from which it was read.
    """
    """ file_list = list(get_path_training_circuits().glob("*.qasm"))

    path_zip = get_path_training_circuits() / "training_data_compilation.zip"
    if len(file_list) == 0 and path_zip.exists():
        with zipfile.ZipFile(str(path_zip), "r") as zip_ref:
            zip_ref.extractall(get_path_training_circuits())

        file_list = list(get_path_training_circuits().glob("*.qasm"))
        assert len(file_list) > 0 """

    #base_path = get_path_training_circuits() / "new_indep_circuits" /"train"
    base_path = get_path_training_circuits() / "training_data_compilation"
    file_list = list(base_path.rglob("*.qasm"))

    """ found_suitable_qc = False
    while not found_suitable_qc:
        random_index = rng.integers(len(file_list))
        num_qubits = int(str(file_list[random_index]).split("_")[-1].split(".")[0])
        if max_qubits and num_qubits > max_qubits:
            continue
        found_suitable_qc = True

    try:
        qc = QuantumCircuit.from_qasm_file(str(file_list[random_index]))
    except Exception:
        raise RuntimeError("Could not read QuantumCircuit from: " + str(file_list[random_index])) from None 
    return qc, str(file_list[random_index])"""
    
    rng = rng or np.random.default_rng()
    while True:
        file_path = rng.choice(file_list)
        try:
            qc = QuantumCircuit.from_qasm_file(str(file_path))
            return qc, str(file_path)
        except Exception:
            print(f"Failed to load {file_path}, retrying...")

    

def get_openqasm_gates() -> list[str]:
    """Returns a list of all quantum gates within the openQASM 2.0 standard header."""
    # according to https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/qasm/libs/qelib1.inc
    return [
        # "u3",
        # "u2",
        # "u1",
        "cx",
        #"id",
        #"u0",
        "u",
        "p",
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "rx",
        "ry",
        "rz",
        "sx",
        "sxdg",
        "cz",
        "cy",
        "swap",
        "ch",
        "ccx",
        "cswap",
        "crx",
        "cry",
        "crz",
        "cu1",
        "cp",
        #"cu3",
        "csx",
        "cu",
        "rxx",
        "rzz",
        "rccx",
        # "rc3x",
        # "c3x",
        # "c3sqrtx",
        # "c4x",
    ]

def dict_to_featurevector(gate_dict: dict[str, int]) -> dict[str, int]:
    """Calculates and returns the feature vector of a given quantum circuit gate dictionary."""
    res_dct = dict.fromkeys(get_openqasm_gates(), 0)
    for key, val in dict(gate_dict).items():
        if key in res_dct:
            res_dct[key] = val

    return res_dct

def create_feature_dict(qc: QuantumCircuit, basis_gates: list[str], coupling_map) -> dict[str, int | NDArray[np.float64]]:
    """Creates a feature dictionary for a given quantum circuit.

    Arguments:
        qc: The quantum circuit for which the feature dictionary is created.

    Returns:
        The feature dictionary for the given quantum circuit.
    """

    gates = get_openqasm_gates()
    ops_list = qc.count_ops()
    ops_list_dict = dict_to_featurevector(ops_list)

    feature_dict = {}
    for key in ops_list_dict:
        feature_dict[key] = float(ops_list_dict[key])
    
    feature_dict["native_mask"] = np.array([1 if g in basis_gates else 0 for g in gates])
    
    feature_dict["measure"] = float(ops_list.get("measure", 0))
    feature_dict["num_qubits"] = float(qc.num_qubits)
    feature_dict["depth"] = float(qc.depth())
    #feature_dict["gate_count"] = float(qc.size())

    
    supermarq_features = calc_supermarq_features(qc)
    # for all dict values, put them in a list each
    feature_dict["program_communication"] = np.array([supermarq_features.program_communication], dtype=np.float32)
    feature_dict["critical_depth"] = np.array([supermarq_features.critical_depth], dtype=np.float32)
    feature_dict["entanglement_ratio"] = np.array([supermarq_features.entanglement_ratio], dtype=np.float32)
    feature_dict["parallelism"] = np.array([supermarq_features.parallelism], dtype=np.float32)
    feature_dict["liveness"] = np.array([supermarq_features.liveness], dtype=np.float32)
    
    
    check_nat_gates = GatesInBasis(basis_gates=basis_gates)
    check_nat_gates(qc)
    only_nat_gates = check_nat_gates.property_set["all_gates_in_basis"]

    check_mapping = CheckMap(coupling_map=CouplingMap(coupling_map))
    check_mapping(qc)
    mapped = check_mapping.property_set["is_swap_mapped"]

    feature_dict["only_nat_gates"] = int(only_nat_gates)
    feature_dict["mapped"] = int(mapped)
    # Graph-based metrics
    """ G = nx.Graph()
    for gate in qc.data:
        if len(gate.qubits) == 2:
            q0 = qc.qubits.index(gate.qubits[0])
            q1 = qc.qubits.index(gate.qubits[1])
            if G.has_edge(q0, q1):
                G[q0][q1]["weight"] += 1
            else:
                G.add_edge(q0, q1, weight=1)

    if len(G.nodes) > 1 and nx.is_connected(G):
        degrees = dict(G.degree())

        feature_dict["avg_hopcount"] = np.array([nx.average_shortest_path_length(G)], dtype=np.float32)
        feature_dict["max_degree"] = max(degrees.values())
        feature_dict["min_degree"] = min(degrees.values())
        #feature_dict["adj_std"] = float(np.std(weights)) if weights else 0.0
    else:
        feature_dict["avg_hopcount"] = np.array([-1.0], dtype=np.float32)
        feature_dict["max_degree"] = 0
        feature_dict["min_degree"] = 0 """
        #feature_dict["adj_std"] = -1.0 

    assert 0 <= feature_dict["num_qubits"] < 128, f'num_qubits {feature_dict["num_qubits"]} out of bounds'

    # depth (Discrete(1_000_000))
    assert 0 <= feature_dict["depth"] < 100_000, f'depth {feature_dict["depth"]} out of bounds'

    # gate_count (Discrete(1_000_000))
    #assert 0 <= feature_dict["gate_count"] < 1_000_00, f'gate_count {feature_dict["gate_count"]} out of bounds'

    # Each openqasm gate in ops_list_dict (Discrete(100_000))
    for gate in ops_list_dict:
        if gate in ['rz', 'sx', 'x', 'cx', 'y', 'z', 'h', 'swap','s', 'sdg', 'u']:
            assert 0 <= feature_dict[gate] < 120_000, f'{gate} count {feature_dict[gate]} out of bounds'
        else:
            assert 0 <= feature_dict[gate] < 6_000, f'{gate} count {feature_dict[gate]} out of bounds'



    # measure (Discrete(10_000))
    assert 0 <= feature_dict["measure"] < 128, f'measure {feature_dict["measure"]} out of bounds'

    """ # avg_hopcount (Box(-1, 128))
    assert -1.0 <= feature_dict["avg_hopcount"] <= 128, f'avg_hopcount {feature_dict["avg_hopcount"]} out of bounds'

    # max_degree (Discrete(128))
    assert 0 <= feature_dict["max_degree"] < 128, f'max_degree {feature_dict["max_degree"]} out of bounds'

    # min_degree (Discrete(128))
    assert 0 <= feature_dict["min_degree"] < 128, f'min_degree {feature_dict["min_degree"]} out of bounds' """

    # program_communication, critical_depth, entanglement_ratio, parallelism, liveness (Box(0,1))
    for field in ["program_communication", "critical_depth", "entanglement_ratio", "parallelism", "liveness"]:
        val = feature_dict[field][0]
        assert 0.0 <= val <= 1.0, f'{field} value {val} out of bounds'

    return feature_dict


def get_path_training_data() -> Path:
    """Returns the path to the training data folder used for RL training."""
    #return Path(str(resources.files("mqt.predictor"))) / "rl" / "training_data"
    return Path(__file__).resolve().parent / "training_data"


def get_path_trained_model() -> Path:
    """Returns the path to the trained model folder used for RL training."""
    return get_path_training_data() / "trained_model"


def get_path_training_circuits() -> Path:
    """Returns the path to the training circuits folder used for RL training."""
    return get_path_training_data() / "training_circuits"


def load_model(model_name: str) -> MaskablePPO:
    """Loads a trained model from the trained model folder.

    Arguments:
        model_name: The name of the model to be loaded.

    Returns:
        The loaded model.
    """
    path = get_path_trained_model()
    if Path(path / (model_name + ".zip")).is_file():
        return MaskablePPO.load(path / (model_name + ".zip"))

    error_msg = f"The RL model '{model_name}' is not trained yet. Please train the model before using it."
    logger.error(error_msg)
    raise FileNotFoundError(error_msg)


def handle_downloading_model(download_url: str, model_name: str) -> None:
    """Downloads a trained model from the given URL and saves it to the trained model folder.

    Arguments:
        download_url: The URL from which the model is downloaded.
        model_name: The name of the model to be downloaded.
    """
    logger.info("Start downloading model...")

    r = requests.get(download_url)
    total_length = int(r.headers.get("content-length"))  # type: ignore[arg-type]
    fname = str(get_path_trained_model() / (model_name + ".zip"))

    with (
        Path(fname).open(mode="wb") as f,
        tqdm(
            desc=fname,
            total=total_length,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in r.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)
    logger.info(f"Download completed to {fname}. ")


class PreProcessTKETRoutingAfterQiskitLayout:
    """Pre-processing step to route a circuit with TKET after a Qiskit Layout pass has been applied.

        The reason why we can apply the trivial layout here is that the circuit already got assigned a layout by qiskit.
        Implicitly, Qiskit is reordering its qubits in a sequential manner, i.e., the qubit with the lowest *physical* qubit
        first.

        Assuming, the layouted circuit is given by

                       ┌───┐           ░       ┌─┐
              q_2 -> 0 ┤ H ├──■────────░───────┤M├
                       └───┘┌─┴─┐      ░    ┌─┐└╥┘
              q_1 -> 1 ─────┤ X ├──■───░────┤M├─╫─
                            └───┘┌─┴─┐ ░ ┌─┐└╥┘ ║
              q_0 -> 2 ──────────┤ X ├─░─┤M├─╫──╫─
                                 └───┘ ░ └╥┘ ║  ║
        ancilla_0 -> 3 ───────────────────╫──╫──╫─
                                          ║  ║  ║
        ancilla_1 -> 4 ───────────────────╫──╫──╫─
                                          ║  ║  ║
               meas: 3/═══════════════════╩══╩══╩═
                                          0  1  2

        Applying the trivial layout, we get the same qubit order as in the original circuit and can be respectively
        routed. This results int:
                ┌───┐           ░       ┌─┐
           q_0: ┤ H ├──■────────░───────┤M├
                └───┘┌─┴─┐      ░    ┌─┐└╥┘
           q_1: ─────┤ X ├──■───░────┤M├─╫─
                     └───┘┌─┴─┐ ░ ┌─┐└╥┘ ║
           q_2: ──────────┤ X ├─░─┤M├─╫──╫─
                          └───┘ ░ └╥┘ ║  ║
           q_3: ───────────────────╫──╫──╫─
                                   ║  ║  ║
           q_4: ───────────────────╫──╫──╫─
                                   ║  ║  ║
        meas: 3/═══════════════════╩══╩══╩═
                                   0  1  2


        If we would not apply the trivial layout, no layout would be considered resulting, e.g., in the followiong circuit:
                 ┌───┐         ░    ┌─┐
       q_0: ─────┤ X ├─────■───░────┤M├───
            ┌───┐└─┬─┘   ┌─┴─┐ ░ ┌─┐└╥┘
       q_1: ┤ H ├──■───X─┤ X ├─░─┤M├─╫────
            └───┘      │ └───┘ ░ └╥┘ ║ ┌─┐
       q_2: ───────────X───────░──╫──╫─┤M├
                               ░  ║  ║ └╥┘
       q_3: ──────────────────────╫──╫──╫─
                                  ║  ║  ║
       q_4: ──────────────────────╫──╫──╫─
                                  ║  ║  ║
    meas: 3/══════════════════════╩══╩══╩═
                                  0  1  2

    """

    def apply(self, circuit: Circuit) -> None:
        """Applies the pre-processing step to route a circuit with tket after a Qiskit Layout pass has been applied."""
        mapping = {Qubit(i): Node(i) for i in range(circuit.n_qubits)}
        place_with_map(circuit=circuit, qmap=mapping)


def get_bqskit_native_gates(device: Device) -> list[gates.Gate] | None:
    """Returns the native gates of the given device.

    Arguments:
        device: The device for which the native gates are returned.

    Returns:
        The native gates of the given provider.
    """
    provider = device.name.split("_")[0]

    native_gatesets = {
        "ibm": [gates.RZGate(), gates.SXGate(), gates.XGate(), gates.CNOTGate()],
        "rigetti": [gates.RXGate(), gates.RZGate(), gates.CZGate()],
        "ionq": [gates.RXXGate(), gates.RZGate(), gates.RYGate(), gates.RXGate()],
        "quantinuum": [gates.RZZGate(), gates.RZGate(), gates.RYGate(), gates.RXGate()],
        "iqm": [gates.U3Gate(), gates.CZGate()],
        "oqc": [gates.RZGate(), gates.XGate(), gates.SXGate(), gates.ECRGate()],
    }

    if provider not in native_gatesets:
        logger.warning("No native gateset for provider " + provider + " found. No native gateset is used.")
        return None

    return native_gatesets[provider]


def final_layout_pytket_to_qiskit(pytket_circuit: Circuit, qiskit_circuit: QuantumCircuit) -> Layout:
    """Converts a final layout from pytket to qiskit."""
    pytket_layout = pytket_circuit.qubit_readout
    size_circuit = pytket_circuit.n_qubits
    qiskit_layout = {}
    qiskit_qreg = qiskit_circuit.qregs[0]

    pytket_layout = dict(sorted(pytket_layout.items(), key=operator.itemgetter(1)))

    for node, qubit_index in pytket_layout.items():
        qiskit_layout[node.index[0]] = qiskit_qreg[qubit_index]

    for i in range(size_circuit):
        if i not in set(pytket_layout.values()):
            qiskit_layout[i] = qiskit_qreg[i]

    return Layout(input_dict=qiskit_layout)


def final_layout_bqskit_to_qiskit(
    bqskit_initial_layout: list[int],
    bqskit_final_layout: list[int],
    compiled_qc: QuantumCircuit,
    initial_qc: QuantumCircuit,
) -> TranspileLayout:
    """Converts a final layout from bqskit to qiskit.

    BQSKit provides an initial layout as a list[int] where each virtual qubit is mapped to a physical qubit
    similarly, it provides a final layout as a list[int] representing where each virtual qubit is mapped to at the end
    of the circuit.
    """
    ancilla = QuantumRegister(compiled_qc.num_qubits - initial_qc.num_qubits, "ancilla")
    qiskit_initial_layout = {}
    counter_ancilla_qubit = 0
    for i in range(compiled_qc.num_qubits):
        if i in bqskit_initial_layout:
            qiskit_initial_layout[i] = initial_qc.qubits[bqskit_initial_layout.index(i)]
        else:
            qiskit_initial_layout[i] = ancilla[counter_ancilla_qubit]
            counter_ancilla_qubit += 1

    initial_qubit_mapping = {bit: index for index, bit in enumerate(compiled_qc.qubits)}

    if bqskit_initial_layout == bqskit_final_layout:
        qiskit_final_layout = None
    else:
        qiskit_final_layout = {}
        for i in range(compiled_qc.num_qubits):
            if i in bqskit_final_layout:
                qiskit_final_layout[i] = compiled_qc.qubits[bqskit_initial_layout[bqskit_final_layout.index(i)]]
            else:
                qiskit_final_layout[i] = compiled_qc.qubits[i]

    return TranspileLayout(
        initial_layout=Layout(input_dict=qiskit_initial_layout),
        input_qubit_mapping=initial_qubit_mapping,
        final_layout=Layout(input_dict=qiskit_final_layout) if qiskit_final_layout else None,
        _output_qubit_list=compiled_qc.qubits,
        _input_qubit_count=initial_qc.num_qubits,
    )


def get_ibm_backend_properties_by_device_name(device_name: str) -> BackendProperties | None:
    """Returns the IBM backend name for the given device name.

    Arguments:
        device_name: The name of the device for which the IBM backend name is returned.

    Returns:
        The IBM backend name for the given device name.
    """
    if "ibm" not in device_name:
        return None
    if device_name == "ibm_washington":
        return FakeWashingtonV2().target
    if device_name == "ibm_montreal":
        return FakeMontrealV2().target
    if device_name == "ibm_guadalupe":
        return FakeGuadalupeV2().target
    if device_name == "ibm_quito":
        return FakeQuitoV2().target
    return None


def postprocess_vf2postlayout(
    qc: QuantumCircuit, post_layout: Layout, layout_before: TranspileLayout
) -> tuple[QuantumCircuit, PassManager]:
    """Postprocesses the given quantum circuit with the post_layout and returns the altered quantum circuit and the respective PassManager."""
    apply_layout = ApplyLayout()
    assert layout_before is not None
    apply_layout.property_set["layout"] = layout_before.initial_layout
    apply_layout.property_set["original_qubit_indices"] = layout_before.input_qubit_mapping
    apply_layout.property_set["final_layout"] = layout_before.final_layout
    apply_layout.property_set["post_layout"] = post_layout

    altered_qc = apply_layout(qc)
    return altered_qc, apply_layout
