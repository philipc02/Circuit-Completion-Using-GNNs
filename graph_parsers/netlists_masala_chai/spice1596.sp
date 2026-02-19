plaintext
* SPICE Netlist for the Differential Amplifier

VDD 3 0 DC 5V
VIN1 1 0 DC 1V
VIN2 2 0 DC 1V

M1 4 1 2 2 NMOS
M2 Vout 2 2 2 NMOS
M3 Vout 3 4 3 PMOS
M4 Vout 3 Vout 3 PMOS

ISS 2 0 DC 1.5mA

* Model Definitions
.model NMOS nmos
.model PMOS pmos

.end