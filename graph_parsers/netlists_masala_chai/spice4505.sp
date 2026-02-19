spice
* Components
R1 Vi 1 R
R2 1 2 R
R3 2 3 R
C1 4 2 3.546C
C2 1 0 1.392C
C3 3 0 0.2024C

* Operational Amplifier (ideal model)
* Vo = A(V+ - V-), with A -> infinity
* Negative input connected to node 3, Positive input to node 2, output to node Vo
EAMP 2 3 Vo 2 100k

* Voltages and Grounds
Vi Vi 0 DC 0
Vgnd 0 0 0

.END