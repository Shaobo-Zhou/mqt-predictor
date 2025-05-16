from pathlib import Path
from qiskit import QuantumCircuit
from mqt.bench.qiskit_helper import get_indep_level as qiskit_indep
from mqt.bench.tket_helper import get_indep_level as tket_indep
import cirq
from qiskit import QuantumCircuit
from cirq.contrib.qasm_import import circuit_from_qasm


from supermarq.benchmarks import (
    bit_code,
    hamiltonian_simulation,
    mermin_bell,
    phase_code,
)


def cirq_to_qiskit(cirq_circuit: cirq.Circuit) -> QuantumCircuit:
    qasm_str = cirq.qasm(cirq_circuit)
    return QuantumCircuit.from_qasm_str(qasm_str)


benchmarks = {
    "bit_code": lambda n: bit_code.BitCode(
        num_data_qubits=n, 
        num_rounds=1, 
        bit_state=[0] * n
    ).circuit(),
    
    "hamiltonian_simulation": lambda n: hamiltonian_simulation.HamiltonianSimulation(
        num_qubits=n, 
        time_step=1, 
        total_time=1
    ).circuit(),

    "mermin_bell": lambda n: mermin_bell.MerminBell(
        num_qubits=n
    ).circuit(),

    "phase_code": lambda n: phase_code.PhaseCode(
        num_data_qubits=n, 
        num_rounds=1, 
        phase_state=[0] * n
    ).circuit(),
}

output_qiskit = Path("compiled_supermarq_qiskit")
output_tket = Path("compiled_supermarq_tket")
output_qiskit.mkdir(parents=True, exist_ok=True)
output_tket.mkdir(parents=True, exist_ok=True)

for name, generator in benchmarks.items():
    for n in range(2, 31):
        try:
            cirq_circ = generator(n)
            qc = cirq_to_qiskit(cirq_circ)
            qc.name = name

            # Qiskit
            qiskit_indep(
                qc, n, file_precheck=False, return_qc=False,
                target_directory=str(output_qiskit)
            )

            # TKET
            tket_indep(
                qc, n, file_precheck=False, return_qc=False,
                target_directory=str(output_tket)
            )

            print(f"✅ Saved {name}-{n}q")

        except Exception as e:
            print(f"⚠️ Failed {name}-{n}q: {e}")
