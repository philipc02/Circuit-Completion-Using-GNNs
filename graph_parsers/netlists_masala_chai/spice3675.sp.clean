plaintext
* NMOS Common Source Amplifier

VDD 3 0 DC 5V
VI 6 0 DC 0V AC 1V

R1 3 1 10k
R2 1 2 10k
RD 3 4 1k

CC 6 1 10pF

M1 4 2 2 2 NMOS L=1u W=10u

.model NMOS NMOS (LEVEL=1 VTO=0.7 KP=50u LAMBDA=0.02)

* Grounding
VSS 2 0 0V

.control
run
.endc