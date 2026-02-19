spice
*MOSFET Transistor Circuit

*.model statement for PMOS and NMOS (not included in the image)
.model PMOS PMOS (LEVEL=1 VTO=-0.7 KP=75u)
.model NMOS NMOS (LEVEL=1 VTO=0.7 KP=100u)

M1 Vout Vin VDD VDD PMOS
M2 Vout Vb 0 0 NMOS

*.end of netlist