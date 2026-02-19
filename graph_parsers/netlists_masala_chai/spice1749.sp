plaintext
* SPICE Netlist
VDD VDD 0 DC <value> ; Define the DC voltage value

Cin 0 input <value> ; Define the capacitance value

M1 VDD input net1 net1 PMOSMODEL ; PMOS transistor
M2 net1 input 0 0 NMOSMODEL ; NMOS transistor

.model PMOSMODEL PMOS (parameters)
.model NMOSMODEL NMOS (parameters)