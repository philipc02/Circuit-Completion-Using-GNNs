* NMOS Amplifier Circuit

V1 6 0 DC 5V
V2 3 0 DC -5V

R1 6 5 14k
R2 5 3 6k
RD 6 2 1.2k
RS 4 3 0.5k

M1 2 5 4 4 NMOS

.model NMOS NMOS level=1

.end