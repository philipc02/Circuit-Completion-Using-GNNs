spice
*MOSFET Parameter Definitions
.model NMOS NMOS LEVEL=1
.model PMOS PMOS LEVEL=1

*Voltage Sources
VDD 4 0 DC 5V
Vb 2 0 DC 1V

*Current Sources
Iin 1 0 DC 1mA

*Resistors
RD 3 4 1k
RF 2 0 1k

*Transistors
M1 1 2 0 0 NMOS
M2 3 2 4 4 PMOS

*End of netlist