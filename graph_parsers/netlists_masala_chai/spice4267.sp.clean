spice
* SPICE Netlist for the given schematic
VPLUS 1 0 DC V+
VMINUS 3 0 DC V-

IREF 1 2 DC IREF
I01 5 0 DC I01
I02 8 0 DC I02
ION 4 0 DC ION

R1 2 6 R1_value

* BJT Q_R (assumed NPN)
QR 6 5 7 NPN_Model

* BJT Q_S (assumed NPN)
QS 2 5 7 NPN_Model

* BJT Q_1 (assumed NPN)
Q1 0 5 3 NPN_Model

* BJT Q_2 (assumed NPN)
Q2 0 8 3 NPN_Model

* BJT Q_N (assumed NPN)
QN 4 4 7 NPN_Model

.model NPN_Model NPN (IS=1e-14 BF=100)

.END