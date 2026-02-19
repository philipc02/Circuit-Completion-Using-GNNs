spice
* SPICE Netlist
V1 1 4 DC Vi
V2 8 0 DC 0

Gmv2 3 2 V2 8 Gm
Gmv1 3 7 V1 6 Gm

Rr2 8 2 Rr2
Ro2 2 3 Ro2
Rr1 1 6 Rr1
Ro1 3 Vo Ro1

* Control Statements
.OP
.END