spice
* SPICE Netlist for the given schematic

* NMOS Transistors
M_REF 2 4 0 0 NMOS
M_1 5 3 0 0 NMOS

* Current Source
I_REF 2 4 DC 1mA

* Resistor
R_B 4 3 1k

* Capacitor
C_B 3 0 1nF

* Voltage Source
V_DD 2 0 DC 5V

* Simulation Commands
*.op
*.end