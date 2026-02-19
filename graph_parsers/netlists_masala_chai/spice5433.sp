* SPICE netlist for the given circuit

V1 1 0 AC 1 SIN(0 1 10k)
L1 1 4 50mH
R1 4 A 5.6k
R2 A 0 3.3k
R3 A B 4.7k
R4 B 4 3.9k
R5 5 0 4.7k
R6 2 0 6.8k

* Analysis
.AC DEC 10 10 10k

.END