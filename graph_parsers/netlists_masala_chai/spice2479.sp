spice
* NMOS Amplifier Circuit

M1 2 Vin 4 4 NMOS_MODEL
RD 2 2 100k
Rgm 4 3 10
VDD 2 0 DC 5V

.model NMOS_MODEL NMOS (LEVEL=1)

.control
tran 1n 100n
.endc

.end