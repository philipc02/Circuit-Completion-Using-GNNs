plaintext
* SPICE Netlist

VCC 7 0 DC 15V
R1 7 9 10k
Q1 9 9 11 NPN
Q2 9 5 11 NPN
Q3 2 2 6 NPN
Q4 2 4 3 NPN
Q5 2 10 4 NPN
IOUT 8 10 DC

* Nodes:
* 0  - Ground
* 2  - Common node for collectors of Q1, Q2, and bases of Q3, Q4, Q5
* 3  - Emitter of Q4
* 4  - Emitter of Q5
* 5  - Base of Q2
* 6  - Emitter of Q3
* 7  - Positive terminal of VCC
* 8  - Common collector node for Q4, Q5 (IOUT)
* 9  - Collector of Q1
* 10 - Collector of Q5 (VOUT)
* 11 - Ground for emitters of Q1, Q2

.END