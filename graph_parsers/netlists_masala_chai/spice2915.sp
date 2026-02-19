* SPICE Netlist for the provided schematic

* Voltage Source
VDD 6 0 DC VDD

* Resistors
R1a 6 3 R1
R1b 9 2 R1
RS 7 5 2RS

* NMOS Transistors (Drain, Gate, Source)
M1 3 8 7 7 NMOS
M2 2 10 5 5 NMOS

* PMOS Transistors (Drain, Gate, Source)
M3 7 4 4 4 PMOS
M4 5 4 4 4 PMOS

* Voltage Inputs
Vin 8 0 DC Vin 
Vb 4 0 DC Vb

* Output
Vout 3 2 0