* RC Low-Pass Filter Netlist
Vin in 0 DC 0 AC 1 SIN(0 1 1k)
R1 in n1 R
C1 n1 0 C
R2 n1 n2 R
C2 n2 0 C
R3 n2 n3 R
C3 n3 0 C
R4 n3 out R
C4 out 0 C
Vout out 0 DC 0

.model R Resistor
.model C Capacitor
.tran 0.1ms 100ms
.end