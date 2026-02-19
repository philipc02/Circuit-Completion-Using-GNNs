spice
* SPICE netlist for the given schematic

*Voltage Source
VCC 5 0 DC 12V

* Resistor
RC 5 2 1k

* Capacitors
Cpi1 2 4 1uF
Cpi2 2 6 1uF
Cmu1 2 5 1uF
Cmu2 5 6 1uF
CCS1 3 0 1uF
CCS2 7 0 1uF

* NPN Transistors
Q1 2 2 3 NPN
Q2 6 2 7 NPN

* Inputs
Vin 4 0 DC 0V
Vb 2 0 DC 1V

.END