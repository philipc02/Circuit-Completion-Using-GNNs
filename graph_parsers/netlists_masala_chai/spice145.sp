spice
* SPICE Netlist for the Given Circuit

* Transistor Models (.model statements are needed for simulation)
*.model PMOS PMOS(L=1u W=1u)
*.model NMOS NMOS(L=1u W=1u)

M1 4 2 6 6 PMOS
M2 33 5 6 6 PMOS
M5 2 3 22 22 NMOS
M6 22 2 2 2 NMOS

* Current Source
IBIAS 6 2 DC 1mA

* Resistors
RL1 4 VDD 1k
RL2 33 VDD 1k

* Voltage Sources (Defined for simulation purpose with power supply values)
VDD VDD 0 DC 5V
VSS VSS 0 DC 0V

* Simulation Commands (Optional, for running the simulation)
*.dc IBIAS 0 1m 0.01m
*.op
*.end