spice
* SPICE Netlist

* MOSFET
M3 5 6 2 2 NMOS

* Resistors
RD 5 6 1k
R1 4 3 1k
R2 3 0 1k

* Voltage Source
VDD 5 0 DC 12V

* Op-Amp
* Assuming ideal op-amp, characteristic internally defined
XOP1 4 4 Y OPAMP

* .model Declaration (Assuming model subcircuit)
.model NMOS NMOS (Level=1)
.subckt OPAMP noninv inv out
+ (subcircuit details, if any)
.ends OPAMP

* Simulation Control
.control
run
.endc

.end