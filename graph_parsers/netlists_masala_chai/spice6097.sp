plaintext
* SPICE netlist for the given circuit

V0 8 3 DC 0
Rx 9 10 1000
Cpi 2 3 1u
Gm 6 3 Vpi 0.01
Ro 7 3 1000
I1 2 3 DC 0

* Vpi is a placeholder for the voltage across Cpi
* Nodes 8, 3 are for V0 (Voltage Source)
* Nodes 9, 10 are for Rx
* Nodes 2, 3 are for Cpi
* Nodes 6, 3 are for the dependent current source
* Nodes 7, 3 are for Ro