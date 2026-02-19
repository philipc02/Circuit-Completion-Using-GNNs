spice
* SPICE Netlist for the provided schematics

Q1 3 5 7 QNPN ; NPN BJT with collector at net 3, base at net 5, and emitter at net 7
Q2 2 3 6 QNPN ; NPN BJT with collector at net 2, base at net 3, and emitter grounded (net 6)

IBIAS 7 8 DC 1mA ; Current source with 1mA between net 7 and ground (net 8)

* Model definitions for the BJTs
.model QNPN NPN (IS=1e-14 BF=100)