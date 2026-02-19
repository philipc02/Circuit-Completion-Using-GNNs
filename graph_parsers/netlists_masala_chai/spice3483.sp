* SPICE Netlist for the Given Circuit

V1 6 7 DC -12
R1 6 4 10k
R2 5 7 1.4k
Q1 2 4 5 NPN
I1 6 2 DC 1mA

* Definition of Nodes
* Node 6: Top of R1, one side of I1, positive terminal of V1
* Node 4: Base of Q1, connection to R1
* Node 2: Collector of Q1, one side of I1
* Node 5: Emitter of Q1, connection to R2, negative terminal of V1
* Node 7: Ground

.END