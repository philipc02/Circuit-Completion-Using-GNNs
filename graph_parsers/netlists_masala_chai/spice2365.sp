* SPICE Netlist
* Components List
C1 2 3 CI
C2 3 5 CIN_BUFFER
C3 3 6 CIN_BUFFER
R1 2 7 R1
R2 3 3 R1
V1 7 0 VIN

* Additional Connections
* Op-Amp (Assumed ideal model)
* Connections: Non-inverting (2), Inverting (3), Output (4)
XOPAMP 2 3 4 OPAMP

.END