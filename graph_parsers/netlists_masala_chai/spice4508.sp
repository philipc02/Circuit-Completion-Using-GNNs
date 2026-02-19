plaintext
* SPICE Netlist
R1 2 5 R/1.082
R2 5 0 R/0.9241
R3 3 8 R/2.613
R4 4 0 R/0.3825

C1 1 2 C
C2 2 3 C
C3 3 4 C

* Voltage Sources and Op-Amps
* Op-amps are modeled with dependent sources in SPICE
* Op-amp 1
E1 5 2 5 2 100k
* Op-amp 2
E2 4 3 4 3 100k

* Input and Output
Vin 1 0 AC 1
Vout 4 0

.end