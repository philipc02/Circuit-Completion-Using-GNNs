plaintext
* SPICE Netlist

VDD 5 0 DC 5V

* Transistors
M1 5 3 0 0 NMOS
M2 2 3 5 5 PMOS

* Resistors
RD 5 2 10k
RF 3 4 10k

* Current Source
Iin 3 0 DC 1mA

* Voltage Source
Vout 4 0 DC 0V

* Simulation
.op
.end