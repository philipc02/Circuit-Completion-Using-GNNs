spice
* SPICE Netlist for the given circuit

Rs 1 4 Rs
Cpi 1 5 Cpi
Rpi 5 6 rpi
Gm 3 4 VOL=V(6,4)
Vx 2 4 Vx
Ix 2 3 Ix

* Voltage measurement for V_pi at node 6 with respect to node 4
Vpi 6 4 DC 0

* End of netlist