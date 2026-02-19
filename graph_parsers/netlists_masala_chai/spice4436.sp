plaintext
* SPICE Netlist

* Voltage Sources
V1 5 0 DC 25
Vs 1 0

* Capacitors
C1 1 4 5u
C2 2 5 5u
C3 3 2 50u
C4 3 0 50u
C5 3 0 10u
C6 6 0 5u

* Resistors
R1 4 0 100
R2 6 0 4.7k
R3 1 2 150k
R4 5 2 10k
R5 1 2 47k
R6 4 0 47k
R7 2 3 33k
R8 3 0 4.7k
R9 0 0 47k

* Transistors
Q1 2 4 3 NPN
Q2 3 5 0 NPN

* Simulation Command
.TRAN 1m 100m
.END