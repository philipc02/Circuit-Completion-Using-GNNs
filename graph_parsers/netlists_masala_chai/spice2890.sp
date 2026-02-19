plaintext
* NMOS Inverter Circuit

M1 2 3 0 0 NMOS

RD 1 2 1k

Cin 3 0 1uF

VDD 1 0 DC 5V

* Node Definitions
* 1: VDD connection
* 2: Output node (Vn1, out)
* 3: Input node (Gate of M1 and Cin)
* 0: Ground

.model NMOS NMOS (Level=1)

.end