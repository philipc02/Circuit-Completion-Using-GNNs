plaintext
* SPICE Netlist

V1 4 0 DC 3V
Rsupply 2 4 46k
Q1 4 3 6 NPN
Q2 3 2 0 PNP
Q3 2 2 0 PNP
Iout 6 0 DC 0A * Current source (I) connected to emitter of Q1
Vin 3 0 DC 0V * Input voltage (vi)

.model NPN npn
.model PNP pnp

.end