spice
* Example SPICE netlist for the circuit

VDD 3 0 DC 5V
VIN1 5 0 DC 1V
VIN2 2 0 DC 1V

M1 6 5 2 2 NMOS L=1u W=1u
M2 6 2 2 2 NMOS L=1u W=1u
M3 3 3 6 6 PMOS L=1u W=1u
M4 2 3 3 3 PMOS L=1u W=1u

ISS 6 0 DC 10uA

*.model NMOS NMOS Level=1
*.model PMOS PMOS Level=1
.end