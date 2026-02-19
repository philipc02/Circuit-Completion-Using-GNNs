plaintext
* SPICE Netlist for the given Schematic

Vsig 6 7 DC 0
Rsig 6 2 Rsig_value
C1 2 3 C1_value
RB 3 0 RB_value
Q1 4 3 5 NPN
I 0 5 DC I_value
VE 5 0 -VEE_value
RC 4 1 RC_value
VCC 1 0 DC VCC_value
C2 1 2 C2_value
RL 2 0 RL_value
CE 5 0 CE_value

.model NPN NPN (IS=1E-14 BF=100)

.end