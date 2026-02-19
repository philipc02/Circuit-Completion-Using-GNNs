spice
* Components
R1 Vin 3 R1
R2 3 2 '2*R1'
R3 3 0 R3
C1 3 2 C
C2 3 2 C

* Voltage source
Vin Vin 0 DC 0

* Operational amplifier
* Assuming an ideal op-amp
* The op-amp has inputs at nodes (3, 2) and output at node 2
E1 2 0 3 2 100k

.END