spice
* SPICE Netlist for the given circuit

V1 1 4 AC 1
V2 5 0 DC 10
V3 8 0 DC -10

R1 5 7 10k
R2 3 7 5k

C1 2 11 CC1
C2 3 0 CC2

Q1 5 11 7 QNPN

.model QNPN NPN (IS=1E-14 BF=100)

.end