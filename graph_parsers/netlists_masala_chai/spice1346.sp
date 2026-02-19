plaintext
* SPICE Netlist for BJT Amplifier Circuit

Q1 N002 Vb N003 NPN

Rout1 N002 0 1k
RE N003 0 100

Vb Vb 0 DC 5

.model NPN NPN (IS=1e-14 BF=100)

.end