spice
* SPICE netlist for the BJT amplifier circuit

VCC 6 0 DC 10

R1 4 3 330k
R2 3 0 100k
RC 6 5 150k
RE 8 0 51

Q1 5 3 8 QNPN

.model QNPN NPN (IS=1e-14 VAF=100 BF=100)

.END