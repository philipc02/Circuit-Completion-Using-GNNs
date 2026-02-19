spice
* SPICE Netlist

VDD 1 0 DC 5V

* Transistors
M1 3 3 0 0 NMOS
M2 2 3 0 0 NMOS
M3 4 4 1 1 PMOS
M4 2 5 4 4 PMOS

* Current Source
ISS 3 0 DC 1mA

* Resistor
RL 2 0 1k

* Nodes:
* 1: VDD
* 2: Output (Vout)
* 3: Connection between M1, M2, and ISS
* 4: Connection between M3, M4, and VDD
* 5: Connection between the drain of M4 and gate of M3

.end