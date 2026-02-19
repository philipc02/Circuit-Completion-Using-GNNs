spice
* SPICE Netlist

* NMOS Transistors
M1 Vout1 Vin+ 3 3 NMOS
M3 Vout2 Vin- 5 5 NMOS
M5 2 Vin+ 6 6 NMOS
M6 2 Vin- 3 3 NMOS

* PMOS Transistors
M2 6 Vout1 2 2 PMOS
M4 2 Vout2 4 4 PMOS
M7 X 6 6 6 PMOS
M8 Y 2 4 4 PMOS

* Voltage Sources
VDD 4 0 DC 5V
Vin+ Vin+ 0 DC 1V
Vin- Vin- 0 DC 0V

* Current Sources
IISS1  5 0 DC 10uA
IISS2  6 0 DC 10uA

* Capacitors
CL1 Vout1 0 1pF
CL2 Vout2 0 1pF

* End of Netlist