spice
* SPICE Netlist
Vgs 1 6 DC 0         ; Voltage source Vgs connected between node 1 and 6
Vx 4 6 DC 0          ; Voltage source Vx connected between node 4 and 6
Gm 7 6 POLY(1) (1,6) (3) 0  ; Dependent current source gm*Vgs between node 7 and 6
R1 7 2 r_o1          ; Resistor ro1 connected between node 7 and 2
R2 2 4 r_o2          ; Resistor ro2 connected between node 2 and 4