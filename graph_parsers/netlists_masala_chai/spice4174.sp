plaintext
* SPICE Netlist

* Current Source
I1 2 4 DC 0 ; Current Source I_S, connected between nodes 2 and 4

* Resistor RS
R1 5 4 RS ; Resistor RS, between nodes 5 and 4

* Resistor Ri
R2 5 3 RI ; Resistor Ri, between nodes 5 and 3

* Resistor RF
R3 3 6 RF ; Resistor RF, between nodes 3 and 6

* Voltage Source (Ground)
V1 4 0 DC 0 ; Voltage source for ground, node 4 to ground

* Operational Amplifier
* The op-amp model is not explicitly defined in standard SPICE, usually requires a subcircuit definition
* The nodes for the op-amp are:
* - Inverting input: node 3
* - Non-inverting input: node 2 (which is grounded)
* - Output: node 2 (v_O)
* A generic op-amp model is used as a placeholder
X1 3 2 2 opamp_sl ; Example subcircuit call

* End of Netlist