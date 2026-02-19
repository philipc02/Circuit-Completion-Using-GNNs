spice
* Components
M1 3 2 2 2 NMOS
RD1 5 3 1k
VDD 5 0 DC 5V
Iin1 3 4 DC 0.1A
Iout1 3 0 DC 0.1A
C1 2 6 1uF

* Node mapping (from annotated image)
* Node 5: VDD
* Node 3: Drain of M1, RD1 connection
* Node 2: Gate and Source of M1
* Node 4: Connection to current source I_n1
* Node 6: Capacitor connection

.MODEL NMOS NMOS(Level=1)
.END