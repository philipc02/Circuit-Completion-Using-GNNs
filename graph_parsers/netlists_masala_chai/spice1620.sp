plaintext
* Differential Amplifier Netlist

.OPTIONS POST=2

* Transistor Models
.MODEL NPN NPN (IS=1E-14 BF=100)

* Nodes
* 1 - Vin1, 2 - Collector of Q3 and Q4, 3 - Base of Q1 and Q2
* 4 - Vout, 5 - IEE, 6 - VCC

* BJTs
Q1 2 1 5 NPN
Q2 2 3 5 NPN
Q3 6 2 3 NPN
Q4 6 4 3 NPN

* Current Source
I1 5 0 DC IEE

* Voltages
VCC 6 0 DC VCC

* Input Voltages
Vin1 1 0 DC 0
Vin2 3 0 DC 0

.END