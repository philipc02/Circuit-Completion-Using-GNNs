plaintext
* Components
V1 8 0 DC 0AC 1 SIN(0 1 1k)
Rsi 8 7 50
Ri 7 4 100k
Cc1 4 2 1u
M1 3 2 6 6 NMOS
M2 9 5 6 6 PMOS
R1 9 3 10k
R2 2 7 10k
Rd1 3 9 10k
Rs1 6 2 5k
Cn1 6 2 20n
Cs 6 0 10u
Rs2 6 5 5k
Cc2 5 0 2p
Ro 5 0 10k
RL 5 0 4k

* Voltage Sources
Vplus 9 0 DC 5
Vminus 2 0 DC -5

* .MODEL Parameters
.model NMOS NMOS(Level=1)
.model PMOS PMOS(Level=1)

* Analysis
.tran 1n 10u
.end