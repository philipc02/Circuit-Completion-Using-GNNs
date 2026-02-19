spice
* Netlist for the given circuit schematic

* NMOS Transistor
M1 X Vin1 0 0 NMOS_MODEL

* PMOS Transistors
M3 3 X 3 3 PMOS_MODEL
M5 3 3 5 3 PMOS_MODEL

* Capacitor
C1 Vout1 0 CL_VALUE

* Resistor
R1 Vout1 5 RD_VALUE

* Voltage Source
VDD 3 0 DC SUPPLY_VALUE

* End of Netlist