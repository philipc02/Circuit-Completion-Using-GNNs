spice
* SPICE Netlist

R1 1 2 30k
R2 2 5 30k

C1 2 4 820p
C2 5 2 1.64n

* Ideal operational amplifier model
* U1 non-inverting input is internally connected
XU1 3 2 5 opamp

V1 4 0 DC 0

* .END statement to terminate the simulation
.end