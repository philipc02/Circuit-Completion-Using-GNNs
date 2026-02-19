plaintext
* Netlist for the given circuit
* Q1 is NMOS and Q2 is PMOS

V_G 3 0 DC <VG_VALUE>
V_i 5 0 DC <VI_VALUE>

* NMOS Device
* Drain Gate Source
M1 4 5 2 2 NMOS

* PMOS Device
* Drain Gate Source
M2 4 3 6 6 PMOS

* Current Source
* Positive terminal first
I1 4 6 DC <IO_VALUE>

* Define the model parameters
.model NMOS nmos (LEVEL=1 VTO=<NMOS_VTO_VALUE> KP=<NMOS_KP_VALUE>)
.model PMOS pmos (LEVEL=1 VTO=<PMOS_VTO_VALUE> KP=<PMOS_KP_VALUE>)

.end