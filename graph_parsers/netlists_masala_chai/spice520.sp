plaintext
*MOSFET Definitions
M1 7 5 6 6 NMOS
M3 3 6 6 6 PMOS

*Voltage Source
V1 5 6 DC vid/2

*Capacitor
C1 4 2 CLd

*Connections
* Node 1 (Net 7) is the Drain of M1
* Node 2 (Net 3) is the Drain of M3
* Node 3 (Net 5) is connected to the gate of M1
* Node 4 (Net 4) is connected to one terminal of CLd
* Node 5 (Net 6) is Ground, connected to source of M1, source and gate of M3