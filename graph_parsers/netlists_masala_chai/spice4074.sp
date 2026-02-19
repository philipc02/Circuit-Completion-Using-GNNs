spice
* Voltage Source
V1 14 13 DC Vi

* Controlled Current Source
G1 5 8 4 0 gm

* Resistors
Rrpi 12 4 rpi
RRE 11 8 RE
RRS 11 14 RS
RRC 9 2 RC
RRL 6 7 RL

* Capacitors
Cmu 4 5 Cmu
Cpi 4 5 Cpi
CL 2 3 CL

* Connections
.nodeset V(B)=10 V(C)=9 V(E)=0