spice
* Circuit Description
R1 3 4 5k
R2 4 2 20k
D1 3 4 D
I1 3 4 DC 0.001
V1 3 0 DC 2
V2 2 0 DC -8

* .model statement for diode
.model D D

* Simulation Commands
.tran 1m 10m
.end