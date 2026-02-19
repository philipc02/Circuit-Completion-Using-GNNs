spice
* SPICE Netlist

I1 6 5 DC {Ii}
R1 7 3 {R1}
C1 3 5 {C1}
G1 5 4 VALUE={gm*v(3,5)}
C2 5 2 {C2}
R2 2 5 {R2}

* Voltage outputs
* v1 is across C1
* vo is across R2

.END