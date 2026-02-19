plaintext
* SPICE Netlist for the given schematic

VCC 6 0 DC 12V

R1 6 5 1k
R2 5 0 1k
R3 5 0 1k

RF_Choke 6 5 10mH
L1 3 2 100uH

C1 5 3 10uF
C2 3 0 10uF
C3 2 4 10uF

Q1 5 5 0 NPN

.END