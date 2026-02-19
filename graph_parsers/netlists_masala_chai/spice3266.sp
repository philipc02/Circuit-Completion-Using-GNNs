spice
* SPICE Netlist for Amplifier Circuit

* NMOS Transistors
M1 X Vin 7 7 NMOS
M2 Y 2 7 7 NMOS

* PMOS Transistors
M3 Vout X 2 2 PMOS
M4 Vout Y 3 3 PMOS
M5 4 6 2 2 PMOS
M6 5 5 2 2 PMOS

* Current Sources
I_M5 6 4 DC 0A
I_M6 5 3 DC 0A
I_SS 7 9 DC 0A

* Voltage Sources
VDD 3 0 DC VDD
Vin Vin 0 DC 0V

* End of Netlist