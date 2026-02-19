spice
* Components
V1 1 2 DC Vin
VB 3 1 DC VB
R1 3 5 1k
D1 5 2 D

* Nets and nodes
* Vin is connected between node 1 (positive) and node 2 (negative)
* VB is connected between node 3 (positive) and node 1 (negative)
* R1 is connected between node 3 and node 5
* D1 is connected between node 5 (anode) and node 2 (cathode)

* Model declaration for diode if necessary
.model D D