spice
* Differential Amplifier Netlist

* Voltage sources
V1 1 4 DC (V_CM + ΔV)
V2 2 5 DC (V_CM - ΔV)

* Current source
I1 3 0 DC I_EE

* Resistor
RL 5 6 RL

* Transistors
Q1 4 3 3 NPN
Q2 5 7 3 NPN

* Load resistor
RL 6 5 RL

* Nodes
* 0: Ground
* 1: V_CM + ΔV Input
* 2: V_CM - ΔV Input
* 3: Common Emitter Node
* 4: Q1 Collector
* 5: Q2 Collector
* 6: Load Resistor
* P: Ground

.END