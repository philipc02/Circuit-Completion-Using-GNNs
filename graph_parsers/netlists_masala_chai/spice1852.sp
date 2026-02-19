spice
* NMOS Amplifier Circuit

M1 4 3 3 3 NMOS
RD 2 4 1k
R1 4 5 10k
R2 5 3 5k
VDD 2 0 DC 5V
VIN 3 0 DC 0V

.model NMOS NMOS(level=1)

.end