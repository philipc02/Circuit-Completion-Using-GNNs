spice
* Components
Vg 9 0 AC 2mV
Vcc 2 0 DC 12V
RG 8 9 50
RE 3 0 2k
R1 4 3 10k
R2 3 6 2k
RC 2 7 3.3k
RL 4 0 10k
C1 8 5 47u
C2 5 3 47u
C3 7 4 1u
Q1 5 3 2 NPN

* Connections
* 1 = Vout
* 2 = Vcc
* 3 = Node between RE, R2, C2
* 4 = Node between R1, RL, C3
* 5 = Node between C1, Q1
* 6 = Node between R2
* 7 = Node between RC and C3
* 8 = Node between RG and C1
* 9 = Node of Vg

* Simulation
.TRAN 1u 10m
.AC DEC 10 1 1MEG
.END