spice
* Spice Netlist
VDD 7 0 DC 5V
Vin 66 6 DC 1V

M5 9 6 8 8 PMOS
M6 3 9 8 8 PMOS
M1 3 8 7 7 PMOS

M3 32 6 2 2 NMOS
M4 3 32 2 2 NMOS

Iss 32 2 DC 1mA

R_Laser 3 5 50
RM 3 4 1k

* End of Netlist